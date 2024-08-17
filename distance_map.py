"""
Created by: Charles Tang
For:        BJ's Wholesale Robotics Team
Summary:    Distance utilities for Forklift VRP 
"""

import itertools
import pandas as pd
import json
import pulp
from pulp import value
from typing import List
from datetime import datetime


def find_distance(dock1: int, dock2: int, dock_map: dict, grid: List[List], type="bfs"):
    """
    Summary: Find distance between two docks
    Inputs:  (int) dock 1 number
             (int) dock 2 number
             (dict) dock # -> (list) 2d coordinate
             (List[List]) grid
    Outputs: (int) Manhattan distance between two docks
    """
    try:
        dock1_coord = dock_map[str(dock1)]
    except:
        raise (f"Dock 1 ({dock1}) does not exist!")
    try:
        dock2_coord = dock_map[str(dock2)]
    except:
        raise (f"Dock 2 ({dock2}) does not exist!")

    # Manhattan distance abs(x1-x2) + abs(y1-y2)
    # replace w A-star or some other fancy algo if needed
    if "bfs":
        return bfs_distance(grid, dock1_coord, dock2_coord)
    return abs(dock1_coord[0] - dock2_coord[0]) + abs(dock1_coord[1] - dock2_coord[1])


def bfs_distance(grid: List[List], start: tuple, end: tuple):
    """
    Summary: Calculates the shortest distance between two dock coordinates in a grid graph using BFS.

    Inputs:
        grid: A 2D list representing the grid graph, where 'X' represents walls.
        start: A tuple (row, col) representing the starting coordinates.
        end: A tuple (row, col) representing the ending coordinates.

    Outputs:
        The shortest distance between start and end coordinates, or -1 if no path exists.
    """

    rows, cols = len(grid), len(grid[0])

    # Check if start and end coordinates are within grid bounds
    if (
        0 <= start[0] < rows
        and 0 <= start[1] < cols
        and 0 <= end[0] < rows
        and 0 <= end[1] < cols
    ):
        # Visited set to track explored cells
        visited = set()
        queue = [(start[0], start[1], 0)]  # (row, col, distance)

        while queue:
            row, col, distance = queue.pop(0)
            if (row, col) == end:
                return distance

            if (row, col) not in visited:
                visited.add((row, col))

                # Explore adjacent cells (up, down, left, right)
                # no need to understand highway directions because it is unintuitive
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_row, new_col = row + dr, col + dc
                    if (
                        0 <= new_row < rows
                        and 0 <= new_col < cols
                        and grid[new_row][new_col] != "X"
                        and (new_row, new_col) not in visited
                    ):
                        queue.append((new_row, new_col, distance + 1))

    return -1  # No path found


def calculate_distance_map(dock_map: dict, grid: List[List]):
    """
    Summary: Find all distance pairs between two docks
    Inputs:  (dict) dock_map where dock # -> (list) 2d coordinate
    Outputs: (dict) (dock1, dock2) -> (int) distance between
    """
    distance_map = {}
    for dock1 in dock_map.keys():
        for dock2 in dock_map.keys():
            if dock1 == dock2:
                continue
            distance = find_distance(dock1, dock2, dock_map, grid, type="bfs")
            distance_map[(dock1, dock2)] = distance
            distance_map[(dock2, dock1)] = distance
    return distance_map


def load_distance_map(dc: int):
    """
    Summary: Load distance map from JSON file
    Inputs:  (int) DC number
    Outputs: (dict) (dock1, dock2) -> (int) distance between
    """
    distance_map = {}
    with open(f"Crossdock Maps/DC{dc}.json", "r", encoding="utf-8") as infile:
        distance_map = json.load(infile)
    new_dist_map = {}
    for x in distance_map:
        values = x[1:-1].split(",")
        new_tuple = (values[0].strip()[1:-1], values[1].strip()[1:-1])
        new_dist_map[new_tuple] = distance_map[x]
    return new_dist_map


def optimize_dock_layout(
    dc: int,
    shipping_doors_counts: List[List],
    method: str = "cluster",
    top_k: int = None,
    middle_index: int = 0,
    calibrationDate: datetime = None,
):
    from data_load import load_aggregate_data

    """
    Create new dist_map by centering shipping doors based on volume around the middle_index.
    Optimize shipping dock door layouts.

    Inputs: dc (int) 800, 820, 840
            shipping_door_counts ([[30, 160], [140, 2000], ...])
            method (str) either 'cluster', 'cluster-spaced', 'greedy', 'linprog'
            top_k (int) number of top shipping doors to consider (only for greedy). Must be less than len(shipping_door_counts) < 2
            middle_index (where to center around; only for cluster)
            calibrationDate (necessary to calibrate greedy model. May be used for linprog)
    Outputs: new_dist_map (dict) updated distance map
             old_new_door_map (dict) swaps in dock doors (old -> new door)
    """
    # Load old dist_map
    dist_map = load_distance_map(dc)

    # Sort by volume
    shipping_doors_counts.sort(key=lambda x: x[1], reverse=True)

    # Sort by shipping door number
    shipping_doors = [x[0] for x in shipping_doors_counts]
    shipping_doors.sort()

    # Provide optimal middle_indices
    match dc:
        case 800:
            middle_index = 54
        case 820:
            middle_index = 44
        case 840:
            middle_index = 34

    old_new_door_map = {}
    match method:
        case "cluster":
            # get middle shipping door
            i = 0
            new_door_order = []
            while i < len(shipping_doors_counts):
                if not top_k or (top_k and i < min(top_k, len(shipping_doors_counts))):
                    if i % 2 == 0:
                        new_door_order.append(shipping_doors_counts[i][0])
                    else:
                        new_door_order.insert(0, shipping_doors_counts[i][0])
                else:
                    new_door_order.append(False)
                i += 1
            # rotate so that shipping_door_counts[i][0] is on middle_index
            while new_door_order[middle_index] != shipping_doors_counts[0][0]:
                new_door_order.append(new_door_order.pop(0))
            # map doors
            for i in range(len(new_door_order)):
                # check if False
                if top_k and new_door_order[i] != False:
                    old_new_door_map[new_door_order[i]] = shipping_doors[i]
                    old_new_door_map[shipping_doors[i]] = new_door_order[i]
                elif not top_k:
                    old_new_door_map[new_door_order[i]] = shipping_doors[i]
        case "cluster-spaced":
            # get middle shipping door
            i = 0
            new_door_order = []
            while i < int(len(shipping_doors_counts) / 2):
                if not top_k or (
                    top_k and i < min(top_k, int(len(shipping_doors_counts) / 2))
                ):
                    # add on left when increasing
                    if i % 2 == 0:
                        new_door_order.append(shipping_doors_counts[-1 - i][0])
                        new_door_order.append(shipping_doors_counts[i][0])
                    else:
                        new_door_order.insert(0, shipping_doors_counts[i][0])
                        new_door_order.insert(0, shipping_doors_counts[-1 - i][0])
                else:
                    new_door_order.append(False)
                i += 1
            # if not all dock doors allocated
            if len(new_door_order) != len(shipping_doors):
                new_door_order.append(
                    shipping_doors_counts[int(len(shipping_doors_counts) / 2) + 1][0]
                )
            # rotate so that shipping_door_counts[i][0] is on middle_index
            while new_door_order[middle_index] != shipping_doors_counts[0][0]:
                new_door_order.append(new_door_order.pop(0))
            # map doors
            for i in range(len(new_door_order)):
                # check if False
                if top_k and new_door_order[i] != False:
                    old_new_door_map[new_door_order[i]] = shipping_doors[i]
                    old_new_door_map[shipping_doors[i]] = new_door_order[i]
                elif not top_k:
                    old_new_door_map[new_door_order[i]] = shipping_doors[i]
        case "greedy":
            # Loop from highest volume to lowest volume
            i = 0
            df = load_aggregate_data(dc=dc, date=calibrationDate)
            # only data in may or jun
            df = df[
                df["from_time"].between("2024-05-22", "2024-05-29", inclusive="both")
            ]
            shipping_doors_assigned = set()
            if not top_k:
                top_k = 100000000
            while i < len(
                shipping_doors_counts[: min(top_k, len(shipping_doors_counts))]
            ):
                original_door = shipping_doors_counts[i][0]
                # Running minimum for minimum distance door
                min_door = None
                min_door_dist = 100000000000
                # Loop through all possible shipping doors
                df_door = df[df["to_locn"] == str(original_door)]  # shipping only
                for door in shipping_doors:
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
                old_new_door_map[original_door] = min_door
                # Remove shipping door from consideration
                shipping_doors_assigned.add(min_door)
                i += 1
            # if top_k, insert swapped locations for doors
            if top_k != 100000000:
                keys = old_new_door_map.keys()
                old_new_door_map_2 = old_new_door_map.copy()
                for door in keys:
                    old_new_door_map_2[old_new_door_map[door]] = door
                old_new_door_map = old_new_door_map_2
        case "linprog":
            # Load in historical data
            df = load_aggregate_data(dc=dc, date=calibrationDate)
            # Only use data in May
            df = df[df["from_time"].dt.month.isin([5, 6])]
            from_locns = sorted(list(set(df["from_locn"])))

            # Params
            num_clubs = len(shipping_doors)
            num_doors = len(shipping_doors)
            num_from_locns = len(from_locns)

            # Create distance matrix: row = dock door -> column = from_locn
            distances = [[0 for i in range(num_from_locns)] for j in range(num_doors)]
            for i in range(num_doors):
                for j in range(num_from_locns):
                    distances[i][j] = dist_map.get(
                        (str(shipping_doors[i]), str(from_locns[j])), 50
                    )

            # Create pallet matrix: number of pallets delivered from column (from_locn) to row (dock_door)
            pallets = [[0 for i in range(num_from_locns)] for j in range(num_doors)]
            pallets_pivot = df.pivot_table(
                index="to_locn", columns="from_locn", aggfunc="size"
            )
            for i in range(num_doors):
                for j in range(num_from_locns):
                    # df_filtered = df[(df["from_locn"] == str(from_locns[j])) & (df["to_locn"] == str(dock_doors[i]))]
                    # pallets[i][j] = df_filtered.shape[0]
                    try:
                        if pd.isnull(
                            pallets_pivot.at[str(shipping_doors[i]), str(from_locns[j])]
                        ):
                            pallets[i][j] = 0
                        else:
                            pallets[i][j] = pallets_pivot.at[
                                str(shipping_doors[i]), str(from_locns[j])
                            ]
                    except:
                        pallets[i][j] = 0

            # Define linear programming problem
            lp_problem = pulp.LpProblem(
                "Shipping_Dock_Door_Allocation", pulp.LpMinimize
            )

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
            sol = lp_problem.solve()

            if pulp.LpStatus[lp_problem.status] == "Optimal":
                optimal_objective = value(lp_problem.objective)
                print(
                    f"Optimal Solution Improvement: {100 * (initial_objective - optimal_objective) / initial_objective}%"
                )
                if top_k:
                    top_k_shipping_doors = set(
                        [
                            _[0]
                            for _ in shipping_doors_counts[
                                : min(top_k, len(shipping_doors_counts))
                            ]
                        ]
                    )
                    top_k_shipping_doors_counts = [
                        _
                        for _ in shipping_doors_counts[
                            : min(top_k, len(shipping_doors_counts))
                        ]
                    ]
                for t in range(num_clubs):
                    for g in range(num_doors):
                        if x[(t, g)].value() == 1:
                            if top_k:
                                if shipping_doors[t] in top_k_shipping_doors:
                                    # find row where door in shipping_doors_counts and add to top_k shipping doors
                                    for i in range(len(shipping_doors_counts)):
                                        if (
                                            shipping_doors_counts[i][0]
                                            == shipping_doors[g]
                                        ):
                                            top_k_shipping_doors_counts.append(
                                                shipping_doors_counts[i]
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
                                old_new_door_map[int(shipping_doors[t])] = int(
                                    shipping_doors[g]
                                )
                if top_k:
                    return optimize_dock_layout(
                        dc=dc,
                        shipping_doors_counts=top_k_shipping_doors_counts,
                        method="linprog",
                        calibrationDate=calibrationDate,
                    )
            else:
                for t in range(num_clubs):
                    old_new_door_map[int(shipping_doors[t])] = int(shipping_doors[t])
    # loop through old_dist_map by keys and create new dist_map
    new_dist_map = {}

    shipping_doors_set = set([str(x) for x in old_new_door_map.keys()])
    for pair in dist_map.keys():
        if pair[0] in shipping_doors_set and pair[1] in shipping_doors_set:
            new_dist_map[
                (
                    # str(old_new_door_map[int(pair[0])]),
                    # str(old_new_door_map[int(pair[1])]),
                    str(pair[0]),
                    str(pair[1]),
                )
            ] = dist_map.get(
                (
                    str(old_new_door_map[int(pair[0])]),
                    str(old_new_door_map[int(pair[1])]),
                ),
                0,
            )
        elif pair[0] in shipping_doors_set:
            new_dist_map[(pair[0], pair[1])] = dist_map.get(
                (
                    str(old_new_door_map[int(pair[0])]),
                    str(pair[1]),
                ),
                0,
            )
        elif pair[1] in shipping_doors_set:
            new_dist_map[pair[0], pair[1]] = dist_map.get(
                (
                    str(pair[0]),
                    str(old_new_door_map[int(pair[1])]),
                ),
                0,
            )
        else:
            new_dist_map[pair] = dist_map[pair]
    return new_dist_map, old_new_door_map
