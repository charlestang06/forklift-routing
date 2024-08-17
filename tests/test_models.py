# Test cases for the models module

import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.curdir))
import pandas as pd
import src.models as models
from src.data_load import get_shipping_doors_volume

# Test data
NUM_FORKLIFTS = 3
START_DATE = datetime.datetime(2021, 1, 1, 0, 0, 0)

distance_map = {}
for i in range(101):
    for j in range(101):
        distance_map[(i, j)] = abs(j - i)
        distance_map[(j, i)] = abs(j - i)

list_of_tasks = [
    [0, 5],
    [1, 6],
    [50, 75],
    [25, 50],
    [100, 25],
    [5, 15],
    [20, 30],
    [95, 85],
    [44, 62],
    [99, 98],
    [98, 33],
    [32, 43],
    [12, 23],
    [23, 43],
    [43, 54],
    [62, 85],
    [100, 6],
    [4, 15],
    [34, 62],
    [35, 62],
    [99, 62],
    [1, 62],
    [9, 23],
    [4, 85],
    [0, 12],
    [93, 62],
]
for i, task_i in enumerate(list_of_tasks):
    time_taken = datetime.timedelta(seconds=distance_map[(task_i[0], task_i[1])])
    list_of_tasks[i] = [
        i,
        START_DATE,
        task_i[0],
        START_DATE + time_taken,
        task_i[1],
        i % NUM_FORKLIFTS,
        0,
    ]
    START_DATE = START_DATE + time_taken

tasks = pd.DataFrame(
    list_of_tasks,
    columns=[
        "id",
        "from_time",
        "from_locn",
        "to_time",
        "to_locn",
        "user",
        "time_taken",
    ],
)

shipping_doors_counts = get_shipping_doors_volume(tasks=tasks)


############### TEST ROUTING MODELS ###############
def test_nearest_neighbor_model():
    """Test Nearest Neighbor Routing Model"""
    model = models.NearestNeighbor(distance_map, tasks, NUM_FORKLIFTS)
    forklift_tasks = model.assign_tasks()
    avg_distance = models.backtest_model(model)
    assert avg_distance < 400
    assert len(forklift_tasks) == NUM_FORKLIFTS


def test_historical_data_model():
    """Test Historical Data"""
    model = models.HistoricalData(distance_map, tasks)
    forklift_tasks = model.assign_tasks()
    avg_distance = models.backtest_model(model)
    assert avg_distance == 476.0
    assert len(forklift_tasks.keys()) == NUM_FORKLIFTS


def test_random_groups_tsp_model():
    """Test Random Groups TSP Routing Model"""
    model = models.RandomGroupsTSP(distance_map, tasks, NUM_FORKLIFTS)
    forklift_tasks = model.assign_tasks()
    avg_distance = models.backtest_model(model)
    assert avg_distance < 400
    assert len(forklift_tasks) == NUM_FORKLIFTS


def test_vrp_model():
    """Test VRP Routing Model"""
    model = models.VRP(distance_map, tasks, NUM_FORKLIFTS)
    forklift_tasks = model.assign_tasks()
    avg_distance = models.backtest_model(model)
    assert avg_distance < 400
    assert len(forklift_tasks) == NUM_FORKLIFTS


def test_compared_results_routing():
    """Test All Model"""
    nn = models.NearestNeighbor(distance_map, tasks, NUM_FORKLIFTS)
    hd = models.HistoricalData(distance_map, tasks, NUM_FORKLIFTS)
    rgtsp = models.RandomGroupsTSP(distance_map, tasks, NUM_FORKLIFTS)
    vrp = models.VRP(distance_map, tasks, NUM_FORKLIFTS)
    dist_nn = models.backtest_model(nn)
    dist_hd = models.backtest_model(hd)
    dist_rgtsp = models.backtest_model(rgtsp)
    dist_vrp = models.backtest_model(vrp)

    assert dist_nn < dist_hd
    assert dist_rgtsp < dist_hd
    assert dist_vrp < dist_hd
    assert dist_nn < dist_rgtsp
    assert dist_vrp < dist_rgtsp
    assert dist_vrp < dist_nn


############### TEST SHIPPING LANE MODELS ###############
def test_cluster_model():
    """Test Clustering Door Model"""
    model = models.ClusterModel(
        tasks=tasks,
        shipping_doors_counts=shipping_doors_counts,
        original_distance_map=distance_map,
    )
    model.assign_doors()
    new_distance_map = model.calculate_distance_map()
    nn = models.NearestNeighbor(new_distance_map, tasks, NUM_FORKLIFTS)
    hd = models.HistoricalData(new_distance_map, tasks)
    dist_nn = models.backtest_model(nn)
    dist_hd = models.backtest_model(hd)
    assert dist_nn < dist_hd


def test_greedy_model():
    """Test Greedy Door Model"""
    model = models.GreedyModel(
        tasks=tasks,
        shipping_doors_counts=shipping_doors_counts,
        original_distance_map=distance_map,
    )
    model.assign_doors()
    new_distance_map = model.calculate_distance_map()
    nn = models.NearestNeighbor(new_distance_map, tasks, NUM_FORKLIFTS)
    hd = models.HistoricalData(new_distance_map, tasks)
    dist_nn = models.backtest_model(nn)
    dist_hd = models.backtest_model(hd)
    assert dist_nn < dist_hd


def test_linprog_model():
    """Test Lin Prog Model"""
    model = models.LinProgModel(
        tasks=tasks,
        shipping_doors_counts=shipping_doors_counts,
        original_distance_map=distance_map,
    )
    model.assign_doors()
    new_distance_map = model.calculate_distance_map()
    nn = models.NearestNeighbor(new_distance_map, tasks, NUM_FORKLIFTS)
    hd = models.HistoricalData(new_distance_map, tasks)
    dist_nn = models.backtest_model(nn)
    dist_hd = models.backtest_model(hd)
    assert dist_nn < dist_hd
