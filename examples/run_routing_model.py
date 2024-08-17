"""
Summary: Guide for running routing model
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
    RandomGroupsTSP,
    VRP,
    backtest_model,
)

# Parameters
DC = 800
DATE = datetime(2024, 6, 1)
NUM_FORKLIFTS = 20
FIRST_X_TASKS = 1000
shipping_doors_counts = get_shipping_doors_volume(
    tasks=load_aggregate_data(dc=DC), dc=DC
)
distance_map = load_distance_map(dc=DC)
tasks = load_aggregate_data(dc=DC, date=DATE).head(FIRST_X_TASKS)

# Instantiate routing models
nn = NearestNeighbor(
    distance_map=distance_map, num_forklifts=NUM_FORKLIFTS, tasks=tasks
)
tsp = RandomGroupsTSP(
    distance_map=distance_map, num_forklifts=NUM_FORKLIFTS, tasks=tasks
)
vrp = VRP(
    distance_map=distance_map,
    num_forklifts=NUM_FORKLIFTS,
    tasks=tasks,
)

# Instantiate historical data model
historical = HistoricalData(distance_map=distance_map, tasks=tasks)

# Test models
nn_dist = backtest_model(model=nn, dc=DC)
tsp_dist = backtest_model(model=tsp, dc=DC)
vrp_dist = backtest_model(model=vrp, dc=DC)
hd_dist = backtest_model(model=historical, dc=DC)

# Print outputs
print(f"Nearest Neighbor Model: {nn_dist}")
print(f"Random Groups TSP Model: {tsp_dist}")
print(f"VRP Model: {vrp_dist}")
print(f"Historical Data: {hd_dist}")
