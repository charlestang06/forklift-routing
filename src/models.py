"""
Summary: Models for routing forklifts
Created by: Charles Tang
For: BJ's Wholesale Robotics Team
"""

from datetime import datetime
import random
from typing import List
import itertools
import numpy as np
import pandas as pd
import pyvrp
from pyvrp.stop import MaxRuntime, NoImprovement, MultipleCriteria
import pulp
from python_tsp.heuristics import solve_tsp_lin_kernighan
from src.distance_map import load_distance_map

pd.options.mode.chained_assignment = None  # default='warn'


############### ROUTING MODELS ####################
class RoutingModel:
    """
    Default class for routing forklifts models. Instantiated for every set of new tasks.
    """

    def __init__(
        self,
        distance_map: dict,
        tasks: pd.DataFrame,
        num_forklifts: int,
        starting_locns=None,
    ):
        """
        Initializes the routing model
        Inputs: distance_map (dict)
                tasks (df)
                num_forklifts (int)
        """
        self.distance_map = distance_map
        self.tasks = tasks
        self.forklifts = num_forklifts
        self.forklift_starting_locns = starting_locns  # may be provided
        self.forklift_tasks = {}

        # Instantiate forklift_tasks
        for i in range(self.forklifts):
            self.forklift_tasks[i] = []

    def assign_tasks(self) -> dict:
        """
        Assigns tasks to forklifts. To be implemented in child classes
        """
        pass

    def get_forklifts_tasks(self) -> dict:
        """
        Returns the tasks assigned to each forklift
        """
        return self.forklift_tasks

    def time_from_distance(self, distance: int) -> float:
        """
        Calculate time (seconds) to travel distance from regression
        Inputs: distance (int)
        """
        return round(37.14 + 0.06 * 6.67 * distance, 0)

    def strip_locn_name(self, name: str) -> str:
        """
        Remove _ and everything after from location name
        """
        if "_" in list(name):
            return name[: name.index("_")]
        return name

    def calcuate_distance_tasks(self, tasks: List[List]) -> int:
        """
        Given list of tasks, calculate distance to travel all tasks
        """
        distance = sum(
            [
                self.distance_map.get((tasks[i][0], tasks[i][1]), 50)
                for i in range(len(tasks))
            ]
        )
        distance += sum(
            [
                self.distance_map.get((tasks[i][1], tasks[i + 1][0]), 50)
                for i in range(len(tasks) - 1)
            ]
        )
        return distance


class HistoricalData(RoutingModel):
    """
    Parse historical data to assign tasks to forklifts.
    """

    def __init__(self, distance_map: dict, tasks: pd.DataFrame, num_forklifts: int):
        super().__init__(distance_map, tasks, len(set(tasks["user"])))

    def assign_tasks(self) -> dict:
        """
        Transform tasks to forklift_tasks format

        Assumptions:
        - Each user in tasks data is a Xdocker
        - Can pick up max 2 pallets
        """
        # Extract list of users
        users = list(set(self.tasks["user"]))
        # Get tasks for each user
        i = 0
        for i in range(self.forklifts):
            user = users[i]
            tasks_user = self.tasks[self.tasks["user"] == user]
            tasks = []
            # Extract as list
            j = 0
            while j < tasks_user.shape[0]:
                row = tasks_user.iloc[j]
                task = [row["from_locn"], row["to_locn"]]
                tasks.append(task)
                # check if next task is the same (double stacking)
                if j < tasks_user.shape[0] - 1:
                    next_row = tasks_user.iloc[j + 1]
                    next_task = [next_row["from_locn"], next_row["to_locn"]]
                    if task == next_task:
                        # add a new task from end of this task to beginning of after skipped task
                        # so # pallets calculation stays same
                        if j < tasks_user.shape[0] - 2:
                            second_next_row = tasks_user.iloc[j + 2]
                            new_task = [
                                next_row["to_locn"],
                                second_next_row["from_locn"],
                            ]
                            tasks.append(new_task)
                        # skip over next task
                        j += 1

                j += 1
            # Add to forklift_tasks dict
            self.forklift_tasks[i] = tasks

        return self.forklift_tasks


class NearestNeighbor(RoutingModel):
    """
    Nearest neighbor/greedy model for routing forklifts with time extrapolation.
    Goal: Minimize distance for each forklift.
    High level: Choose task based on distance to current location.
    Speed: Very fast, not optimal.
    """

    def assign_tasks(self) -> dict:
        """
        Implementation of the nearest neighbor model

        Assumptions:
        - Up to 1 pallet at a time
        - All tasks are same priority
        """

        time = 0
        task_done_times = [0 for i in range(self.forklifts)]

        # Aggregate tasks
        tasks = []  # [["111", "114"], ["631", "91"], ...]
        for i in range(self.tasks.shape[0]):
            row = self.tasks.iloc[i]
            tasks.append([row["from_locn"], row["to_locn"]])
        random.shuffle(tasks)

        # Assign first task for each forklift
        if self.forklift_starting_locns:
            # Loop through each forklift
            for i in range(self.forklifts):
                curr_locn = self.forklift_starting_locns[i]
                closest_distance = 10000000
                closest_index = -1
                if len(tasks) == 0:
                    break
                # Find closest task
                for j, task in enumerate(tasks):
                    task = tasks[j]
                    distance = self.distance_map.get((task[0], curr_locn), 100000000)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_index = j
                closest_task = tasks.pop(closest_index)
                # Add task from curr_locn to starting task
                self.forklift_tasks[i].append([curr_locn, closest_task[0]])
                # Add closest task to forklift tasks dict
                self.forklift_tasks[i].append(closest_task)
                # Set task done time for this forklift
                task_done_times[i] = self.time_from_distance(  # type: ignore
                    closest_distance
                ) + self.time_from_distance(
                    self.distance_map.get((closest_task[0], closest_task[1]), 50)
                )

        # Starting locns not provided -> start at random task
        else:
            for i in range(self.forklifts):
                if len(tasks) == 0:
                    break
                task = tasks.pop(0)
                self.forklift_tasks[i].append(task)
                task_done_times[i] = self.time_from_distance(  # type: ignore
                    self.distance_map.get((task[0], task[1]), 50)
                )

        # Simulate with nearest neighbor heuristic until all tasks completed
        while len(tasks) > 0:
            time += 1
            # Loop through end times if any forklift done with task
            for i in range(self.forklifts):
                if len(tasks) == 0:
                    break
                # If forklift finished task
                if task_done_times[i] <= time:
                    # Last locn of the last task
                    curr_locn = self.forklift_tasks[i][-1][1]
                    closest_distance = 10000000
                    closest_index = -1
                    # Find closest task
                    for j, task in enumerate(tasks):
                        task = tasks[j]
                        distance = self.distance_map.get(
                            (task[0], curr_locn), 100000000
                        )
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_index = j
                    closest_task = tasks.pop(closest_index)
                    # Add closest task to forklift tasks dict
                    self.forklift_tasks[i].append(closest_task)
                    # Set task done time for this forklift
                    task_done_times[i] = self.time_from_distance(  # type: ignore
                        closest_distance
                    ) + self.time_from_distance(
                        self.distance_map.get((closest_task[0], closest_task[1]), 50)
                    )
        return self.forklift_tasks


class RandomGroupsTSP(RoutingModel):
    """
    Assign tasks to random groups and route forklifts TSPPD.
    Goal: Minimize total distance. Assume only one pallet at a time.
    High level: Linear programming problem.
    Speed: Very fast, near optimal.
    """

    def assign_tasks(self) -> dict:
        """
        Assign tasks randomly
        """
        # Aggregate tasks in better format
        tasks = []  # [["111", "114"], ["631", "91"], ...]
        for i in range(self.tasks.shape[0]):
            row = self.tasks.iloc[i]
            tasks.append([row["from_locn"], row["to_locn"]])
        random.shuffle(tasks)

        # Partition tasks (chunking)
        for item in tasks:
            min_size_chunk_index = min(
                range(self.forklifts), key=lambda i: len(self.forklift_tasks[i])
            )
            self.forklift_tasks[min_size_chunk_index].append(item)

        # TSP Orderings for each forklift tasks
        for key, item in self.forklift_tasks.items():
            # Construct distance matrix
            distance_matrix = [
                [0 for _ in range(len(tasks))] for _ in range(len(tasks))
            ]
            for i, task_i in enumerate(tasks):
                for j, task_j in enumerate(tasks):
                    if i == j:
                        distance_matrix[i][j] = 0
                    elif task_i[1] == task_j[0]:
                        distance_matrix[i][j] = 0
                    else:
                        # from task i to task j
                        distance_matrix[i][j] = self.distance_map[
                            (task_i[1], task_j[0])
                        ]
            distance_matrix_arr = np.array(distance_matrix)
            distance_matrix_arr[:, 0] = 0  # depot

            # Run TSP
            permutation, _ = solve_tsp_lin_kernighan(distance_matrix_arr)

            new_tasks = []
            for i, index in enumerate(permutation):
                index = permutation[i]
                new_tasks.append(tasks[index])

            self.forklift_tasks[key] = new_tasks
        return self.forklift_tasks


class VRP(RoutingModel):
    """
    Use PyVRP for CVRPPD. Assigns each task as a "city".
    Goal: Minimize total distance. Assume only one pallet at a time.
    High level: VRP problem.
    Speed: Very fast, near optimal.
    """

    def assign_tasks(self) -> dict:
        """
        Assign tasks with VRP in order optimally
        """
        # Aggregate tasks in better format
        tasks = []  # [["111", "114"], ["631", "91"], ...]
        for i in range(self.tasks.shape[0]):
            row = self.tasks.iloc[i]
            tasks.append([row["from_locn"], row["to_locn"]])
        random.shuffle(tasks)

        # Create distance matrix
        distance_matrix = [[0 for _ in range(len(tasks))] for _ in range(len(tasks))]
        for i, task_i in enumerate(tasks):
            for j, task_j in enumerate(tasks):
                if i == j:
                    distance_matrix[i][j] = 0
                elif task_i[1] == task_j[0]:
                    distance_matrix[i][j] = 0
                else:
                    # from task i to task j
                    distance_matrix[i][j] = self.distance_map.get(
                        (task_i[1], task_j[0]), 50
                    )
        distance_matrix = np.array(distance_matrix)
        # Add row of 0s (fake depot)
        zero_row = np.zeros(distance_matrix.shape[1])
        distance_matrix = np.vstack([zero_row, distance_matrix])
        # Add col of 0s (fake depot)
        zero_col = np.zeros((distance_matrix.shape[0], 1))
        distance_matrix = np.hstack([zero_col, distance_matrix])
        distance_matrix = distance_matrix.tolist()
        for i, _ in enumerate(distance_matrix):
            for j, _ in enumerate(distance_matrix):
                distance_matrix[i][j] = int(distance_matrix[i][j])

        # Instantiate model and create depot/clients
        m = pyvrp.Model()
        m.add_vehicle_type(
            self.forklifts, capacity=int(len(tasks) / self.forklifts) + 2
        )
        depot = m.add_depot(x=0, y=0)
        clients = [
            m.add_client(x=0, y=0, delivery=1) for idx in range(1, len(distance_matrix))
        ]

        # Add edges based on distance matrix
        locations = [depot] + clients
        for i, _ in enumerate(locations):
            for j, _ in enumerate(locations):
                distance = distance_matrix[i][j]
                m.add_edge(locations[i], locations[j], distance=distance)

        # Solve VRP
        stopping = MultipleCriteria(criteria=[MaxRuntime(20), NoImprovement(10)])
        res = m.solve(stop=stopping, display=False)  # 10 seconds or no improvement
        routes = []
        for x in res.best.routes():
            route = str(x)
            routes.append(list(map(int, route.split())))

        # Replace route indices with task numbers
        i = 0
        for key in self.forklift_tasks:
            route = routes[i]
            for j, route_j in enumerate(route):
                idx = route_j - 1
                route[j] = tasks[idx]
            i += 1
            self.forklift_tasks[key] = route

        return self.forklift_tasks


############### SHIPPING LANE CONFIG MODELS ########################
class ShippingLaneModel:
    """
    Default class for assigning dock doors. Instantiated for every DC and method combination.
    """

    def __init__(
        self,
        tasks: pd.DataFrame,
        shipping_doors_counts: List[List],
        original_distance_map: dict,
        dc: int = None,  # type: ignore
        top_k: int = None,  # type: ignore
        calibration_date: datetime = None,  # type: ignore
    ):
        """
        Initializes the shipping lane config model
        Inputs: tasks (Dataframe)
                shipping_doors_counts (List[List])
                original_distance_map (dict)
                dc (int)
                top_k (int)
                calibrationDate (datetime)
        """
        self.dc = dc
        self.tasks = tasks
        self.shipping_doors_counts = shipping_doors_counts
        self.shipping_doors_counts.sort(key=lambda x: x[1], reverse=True)
        # Sort by shipping door number
        self.shipping_doors = [x[0] for x in self.shipping_doors_counts]
        self.shipping_doors.sort()
        self.top_k = top_k
        self.calibration_date = calibration_date
        self.original_distance_map = original_distance_map
        self.old_new_door_map = {}
        self.new_distance_map = {}

        # for clustering models
        match self.dc:
            case 800:
                self.middle_index = 54
            case 820:
                self.middle_index = 44
            case 840:
                self.middle_index = 34
            case _:
                self.middle_index = int(len(shipping_doors_counts) / 2)

    def assign_doors(self):
        """
        Assign doors parent function
        """
        pass

    def calculate_distance_map(self) -> dict:
        """
        Calculates new distance map given the newly assigned locations for each dock door
        """
        # loop through old_dist_map by keys and create new dist_map
        new_dist_map = {}
        dist_map = self.original_distance_map

        shipping_doors_set = set([str(x) for x in self.old_new_door_map])
        for pair in dist_map.keys():
            if pair[0] in shipping_doors_set and pair[1] in shipping_doors_set:
                new_dist_map[
                    (
                        str(pair[0]),
                        str(pair[1]),
                    )
                ] = dist_map.get(
                    (
                        str(self.old_new_door_map[int(pair[0])]),
                        str(self.old_new_door_map[int(pair[1])]),
                    ),
                    0,
                )
            elif pair[0] in shipping_doors_set:
                new_dist_map[(pair[0], pair[1])] = dist_map.get(
                    (
                        str(self.old_new_door_map[int(pair[0])]),
                        str(pair[1]),
                    ),
                    0,
                )
            elif pair[1] in shipping_doors_set:
                new_dist_map[pair[0], pair[1]] = dist_map.get(
                    (
                        str(pair[0]),
                        str(self.old_new_door_map[int(pair[1])]),
                    ),
                    0,
                )
            else:
                new_dist_map[pair] = dist_map[pair]

        self.new_distance_map = new_dist_map
        return new_dist_map


class ClusterModel(ShippingLaneModel):
    """
    Class for assigning dock doors by clustering highest volume dock doors.
    """

    def assign_doors(self) -> dict:
        """
        Create old_new_door_map for assigning old door to new door
        """
        # get middle shipping door
        i = 0
        new_door_order = []
        while i < len(self.shipping_doors_counts):
            if not self.top_k or (
                self.top_k and i < min(self.top_k, len(self.shipping_doors_counts))
            ):
                if i % 2 == 0:
                    new_door_order.append(self.shipping_doors_counts[i][0])
                else:
                    new_door_order.insert(0, self.shipping_doors_counts[i][0])
            else:
                new_door_order.append(False)
            i += 1
        # rotate so that shipping_door_counts[i][0] is on middle_index
        while new_door_order[self.middle_index] != self.shipping_doors_counts[0][0]:
            new_door_order.append(new_door_order.pop(0))
        # map doors
        for i, new_door_order_i in enumerate(new_door_order):
            # check if False
            if self.top_k and new_door_order_i is not False:
                self.old_new_door_map[new_door_order_i] = self.shipping_doors[i]
                self.old_new_door_map[self.shipping_doors[i]] = new_door_order_i
            elif not self.top_k:
                self.old_new_door_map[new_door_order_i] = self.shipping_doors[i]

        return self.old_new_door_map


class GreedyModel(ShippingLaneModel):
    """
    Class for assigning dock doors by assigning highest volume dock doors in greedy method.
    """

    def assign_doors(self) -> dict:
        """
        Create old_new_door_map for assigning old door to new door
        """

        # Loop from highest volume to lowest volume
        i = 0
        df = self.tasks
        dist_map = self.original_distance_map
        shipping_doors_assigned = set()
        if not self.top_k:
            self.top_k = 100000000
        while i < len(
            self.shipping_doors_counts[
                : min(self.top_k, len(self.shipping_doors_counts))
            ]
        ):
            original_door = self.shipping_doors_counts[i][0]
            # Running minimum for minimum distance door
            min_door = None
            min_door_dist = 100000000000
            # Loop through all possible shipping doors
            df_door = df[df["to_locn"] == str(original_door)]  # shipping only
            for door in self.shipping_doors:
                if door in shipping_doors_assigned:
                    continue

                # total_dist = # pallets to original door * dist to new door
                def calculate_distance(row):
                    return dist_map.get((row["from_locn"], str(door)), 50)

                df_door["distance"] = df.apply(calculate_distance, axis=1)
                door_dist = sum(df_door["distance"])
                if door_dist < min_door_dist:
                    min_door_dist = door_dist
                    min_door = door
            # Map new door -> old door locn
            self.old_new_door_map[original_door] = min_door
            # Remove shipping door from consideration
            shipping_doors_assigned.add(min_door)
            i += 1
        # if top_k, insert swapped locations for doors
        if self.top_k != 100000000:
            keys = self.old_new_door_map.keys()
            old_new_door_map_2 = self.old_new_door_map.copy()
            for door in keys:
                old_new_door_map_2[self.old_new_door_map[door]] = door
            self.old_new_door_map = old_new_door_map_2

        return self.old_new_door_map


class LinProgModel(ShippingLaneModel):
    """
    Class for assigning dock doors by using Linear Programming (Stankovic et al., 2022).
    """

    def assign_doors(self) -> dict:
        """
        Create old_new_door_map for assigning old door to new door
        """
        # Load in historical data
        df = self.tasks
        # Only use data in May
        from_locns = sorted(list(set(df["from_locn"])))

        dist_map = self.original_distance_map

        # Params
        num_clubs = len(self.shipping_doors)
        num_doors = len(self.shipping_doors)
        num_from_locns = len(from_locns)

        # Create distance matrix: row = dock door -> column = from_locn
        distances = [[0 for i in range(num_from_locns)] for j in range(num_doors)]
        for i in range(num_doors):
            for j in range(num_from_locns):
                distances[i][j] = dist_map.get(
                    (str(self.shipping_doors[i]), str(from_locns[j])), 50
                )

        # Create pallet matrix: number of pallets delivered from column (from_locn) to row (dock_door)
        pallets = [[0 for i in range(num_from_locns)] for j in range(num_doors)]
        pallets_pivot = df.pivot_table(
            index="to_locn", columns="from_locn", aggfunc="size"
        )
        for i in range(num_doors):
            for j in range(num_from_locns):
                try:
                    if pd.isnull(
                        pallets_pivot.at[
                            str(self.shipping_doors[i]), str(from_locns[j])
                        ]
                    ):
                        pallets[i][j] = 0
                    else:
                        pallets[i][j] = pallets_pivot.at[
                            str(self.shipping_doors[i]), str(from_locns[j])
                        ]
                except KeyError:
                    pallets[i][j] = 0

        # Define linear programming problem
        lp_problem = pulp.LpProblem("Shipping_Dock_Door_Allocation", pulp.LpMinimize)

        # Define decision variables
        x = {}
        for t in range(num_clubs):
            for g in range(num_doors):
                x[(t, g)] = pulp.LpVariable(f"x_{t}_{g}", cat="Binary")

        # Define the objective function
        lp_problem += pulp.lpSum(
            x[(t, g)] * distances[g][z] * pallets[t][z]
            for t in range(num_clubs)
            for g in range(num_doors)
            for z in range(num_from_locns)
        )

        # Calculate initial objective value
        initial_objective = 0
        for t in range(num_clubs):
            for g in range(num_doors):
                for z in range(num_from_locns):
                    if t == g:
                        initial_objective += distances[g][z] * pallets[t][z]

        # Define constraints
        # Each club must be assigned to a dock_door
        for t in range(num_clubs):
            lp_problem += pulp.lpSum(x[(t, g)] for g in range(num_doors)) == 1

        # Each gate can be assigned to only one or no trucks
        for g in range(num_doors):
            lp_problem += pulp.lpSum(x[(t, g)] for t in range(num_clubs)) <= 1

        # Solve the linear programming problem
        lp_problem.solve()

        if pulp.LpStatus[lp_problem.status] == "Optimal":
            if self.top_k:
                top_k_shipping_doors = set(
                    [
                        _[0]
                        for _ in self.shipping_doors_counts[
                            : min(self.top_k, len(self.shipping_doors_counts))
                        ]
                    ]
                )
                top_k_shipping_doors_counts = [
                    _
                    for _ in self.shipping_doors_counts[
                        : min(self.top_k, len(self.shipping_doors_counts))
                    ]
                ]
            for t in range(num_clubs):
                for g in range(num_doors):
                    if x[(t, g)].value() == 1:
                        if self.top_k:
                            if self.shipping_doors[t] in top_k_shipping_doors:
                                # find row where door in shipping_doors_counts and add to top_k shipping doors
                                for i, shipping_doors_counts_i in enumerate(
                                    self.shipping_doors_counts
                                ):
                                    if (
                                        shipping_doors_counts_i[0]
                                        == self.shipping_doors[g]
                                    ):
                                        top_k_shipping_doors_counts.append(
                                            shipping_doors_counts_i
                                        )
                                        break
                                top_k_shipping_doors_counts.sort()
                                list(
                                    k
                                    for k, _ in itertools.groupby(
                                        top_k_shipping_doors_counts
                                    )
                                )
                        else:
                            self.old_new_door_map[int(self.shipping_doors[t])] = int(
                                self.shipping_doors[g]
                            )
            if self.top_k:
                linprogmodel = LinProgModel(
                    tasks=self.tasks,
                    dc=self.dc,
                    shipping_doors_counts=top_k_shipping_doors_counts,
                    original_distance_map=self.original_distance_map,
                    top_k=None,  # type: ignore
                    calibration_date=self.calibration_date,
                )
                self.old_new_door_map = linprogmodel.assign_doors()
        else:
            for t in range(num_clubs):
                self.old_new_door_map[int(self.shipping_doors[t])] = int(
                    self.shipping_doors[t]
                )

        return self.old_new_door_map


class ClusterSpacedModel(ShippingLaneModel):
    """
    Class to assign middle-volume clubs to space apart high-volume clubs and cluster near the center.
    """

    def assign_doors(self) -> dict:
        """
        Create old_new_door_map for assigning old door to new door and spacing with middle-volume doors
        """
        # get middle shipping door
        i = 0
        new_door_order = []
        middle_index = int(len(self.shipping_doors_counts) / 2)
        while i < len(self.shipping_doors_counts):
            if not self.top_k or (
                self.top_k
                and int(i / 2) < min(self.top_k, len(self.shipping_doors_counts))
            ):
                if i == len(self.shipping_doors_counts) - 1:
                    new_door_order.append(self.shipping_doors_counts[i][0])
                elif int(i / 2) % 2 == 0:
                    new_door_order.append(self.shipping_doors_counts[int(i / 2)][0])
                    new_door_order.append(
                        self.shipping_doors_counts[middle_index + int(i / 2)][0]
                    )
                else:
                    new_door_order.insert(
                        0, self.shipping_doors_counts[middle_index + int(i / 2)][0]
                    )
                    new_door_order.insert(0, self.shipping_doors_counts[int(i / 2)][0])
                i += 2
            else:
                new_door_order.append(False)
                i += 1
        # rotate so that shipping_door_counts[i][0] is on middle_index
        while new_door_order[self.middle_index] != self.shipping_doors_counts[0][0]:
            new_door_order.append(new_door_order.pop(0))
        # map doors
        for i, new_door_order_i in enumerate(new_door_order):
            # check if False
            if self.top_k and new_door_order_i is not False:
                self.old_new_door_map[new_door_order_i] = self.shipping_doors[i]
                self.old_new_door_map[self.shipping_doors[i]] = new_door_order_i
            elif not self.top_k:
                self.old_new_door_map[new_door_order_i] = self.shipping_doors[i]
        return self.old_new_door_map


class ManualOldNewDoorMap(ShippingLaneModel):
    """
    Custom defined old_new_door_map
    """

    def __init__(self, old_new_door_map: dict, *args, **kwargs):
        """
        Initializes the shipping lane config model
        Inputs: old_new_door_map (dict)
                **other ShippingLaneModel args
        """
        super().__init__(*args, **kwargs)
        self.old_new_door_map = old_new_door_map

    def assign_doors(self) -> dict:
        return self.old_new_door_map


############## TESTING ROUTING MODEL #########################
def backtest_model(model: RoutingModel, dc: int = None) -> float:  # type: ignore
    """
    Loads in RoutingModel and backtests on a historical tasks
    Inputs: model - RoutingModel
    Outputs: avg_distance - average distance per pallet averaged across all Xdockers
    """
    avg_distances = []
    total_distances = []
    total_pallets = 0

    # Allocate routes
    routes = model.assign_tasks()

    # Load dist_map if not provided
    dist_map = model.distance_map
    if not dist_map and dc:
        dist_map = load_distance_map(dc)

    # Loop through each user
    for user in routes.keys():
        dist = 0
        route = routes[user]
        num_pallets = len(route)
        total_pallets += num_pallets
        # Loop through user's route
        for i, route_i in enumerate(route):
            # task distance
            dist += dist_map.get((route_i[0], route_i[1]), 30)
            if i < len(route) - 1:
                # distance to next task
                dist += dist_map.get((route_i[1], route[i + 1][0]), 30)
        # Add to data
        if num_pallets > 0:
            avg_distances.append(round(dist / num_pallets, 0))
            total_distances.append(dist)

    # Average of total distances
    return round(sum(total_distances) / total_pallets * 6.67, 0)
