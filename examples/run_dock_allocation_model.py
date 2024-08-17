"""
Summary: Guide for running dock door allocation model
Created by: Charles Tang
For: BJ's Wholesale Robotics Team
"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.curdir))

from src.data_load import load_aggregate_data, get_shipping_doors_volume
from src.distance_map import load_distance_map
from src.models import (
    NearestNeighbor,
    HistoricalData,
    ClusterModel,
    GreedyModel,
    LinProgModel,
    backtest_model,
)

# Parameters
DC = 840
DATE = datetime(2024, 6, 1)
NUM_FORKLIFTS = 20
FIRST_X_TASKS = 1000
tasks = load_aggregate_data(dc=DC, date=DATE).head(FIRST_X_TASKS)
all_tasks = load_aggregate_data(dc=DC)
shipping_doors_counts = get_shipping_doors_volume(tasks=all_tasks, dc=DC)
distance_map = load_distance_map(dc=DC)

# Cluster model
cluster = ClusterModel(
    tasks=all_tasks,
    dc=DC,
    shipping_doors_counts=shipping_doors_counts,
    original_distance_map=distance_map,
    top_k=None,  # type: ignore
    calibration_date=None,  # type: ignore
)
cluster_door_map = cluster.assign_doors()
cluster_distance_map = cluster.calculate_distance_map()
routingCluster = NearestNeighbor(
    distance_map=cluster_distance_map,
    num_forklifts=20,
    tasks=tasks,
)
clusterDistance = backtest_model(model=routingCluster, dc=DC)

# Greedy model
greedy = GreedyModel(
    tasks=tasks,
    dc=DC,
    shipping_doors_counts=shipping_doors_counts,
    original_distance_map=distance_map,
    top_k=None,  # type: ignore
    calibration_date=DATE,  # Greedy model is slow so needs calibrationDate
)
greedy_door_map = greedy.assign_doors()
greedy_distance_map = greedy.calculate_distance_map()
routingGreedy = NearestNeighbor(
    distance_map=greedy_distance_map,
    num_forklifts=20,
    tasks=tasks,
)
greedyDistance = backtest_model(model=routingGreedy, dc=DC)

# Linear Programming Model
linprog = LinProgModel(
    tasks=all_tasks,
    dc=DC,
    shipping_doors_counts=shipping_doors_counts,
    original_distance_map=distance_map,
    top_k=None,  # type: ignore
    calibration_date=None,  # Greedy model is slow so needs calibrationDate # type: ignore
)
linprog_door_map = linprog.assign_doors()
linprog_distance_map = linprog.calculate_distance_map()
routingLinProg = NearestNeighbor(
    distance_map=linprog_distance_map,
    num_forklifts=20,
    tasks=tasks,
)
linProgDistance = backtest_model(model=routingLinProg, dc=DC)

# Historical model
historical = HistoricalData(
    distance_map=distance_map,
    tasks=tasks,
)
historicalDistance = backtest_model(model=historical, dc=DC)

# Print outputs
print(f"Clustering Model: {clusterDistance}")
print(f"Greedy Model: {greedyDistance}")
print(f"LinProg Model: {linProgDistance}")
print(f"Historical Data: {historicalDistance}")
