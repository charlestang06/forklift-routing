"""
Summary: Test for comparing model performance with historical/original data
Created by: Charles Tang
For: BJ's Wholesale Robotics Team
"""

from datetime import datetime
from data_load import load_aggregate_data, get_shipping_doors_volume
from gridload import bfs_path, load_grid_graph, plot_grid
from distance_map import load_distance_map, optimize_dock_layout
from models import (
    NearestNeighbor,
    HistoricalData,
    RandomGroupsTSP,
    VRP,
    ClusterModel,
    GreedyModel,
    LinProgModel,
    ClusterSpacedModel,
    ManualOldNewDoorMap,
)
from test import backtest_model

################## TEST DOORS ######################
dc = 820
shipping_doors_counts = get_shipping_doors_volume(dc=dc)
old_distance_map = load_distance_map(dc=dc)
# door_model = ClusterSpacedModel(
#     dc=dc,
#     shipping_doors_counts=shipping_doors_counts,
#     original_distance_map=old_distance_map,
#     top_k=5,
# )
old_new_door_map = {
    58: 106,
    106: 110,
    110: 58,
    82: 82,
    165: 84,
    84: 165,
    138: 86,
    86: 138,
    52: 88,
    88: 52,
    172: 108,
    108: 172,
    36: 112,
    112: 36,
    66: 114,
    114: 66,
    28: 116,
    116: 28,
}
door_model = ManualOldNewDoorMap(
    dc=dc,
    shipping_doors_counts=shipping_doors_counts,
    old_new_door_map=old_new_door_map,
    original_distance_map=old_distance_map,
)
old_new_door_map = door_model.assign_doors()
print(old_new_door_map)
new_distance_map = door_model.calculate_distance_map()
improvements = []
avg_dist_bad = []
avg_dist_good = []
for day in range(15, 42):
    if day > 30:
        date = datetime(2024, 6, day % 30)
    else:
        date = datetime(2024, 5, day)
    tasks = load_aggregate_data(dc=dc, date=date).head(1000)
    routingOldShippingDocks = VRP(
        distance_map=old_distance_map,
        num_forklifts=20,
        tasks=tasks,
    )
    oldDistance = backtest_model(routingOldShippingDocks, dc=dc)
    avg_dist_bad.append(oldDistance)
    routingNewShippingDocks = VRP(
        distance_map=new_distance_map,
        num_forklifts=20,
        tasks=tasks,
    )
    newDistance = backtest_model(routingNewShippingDocks, dc=dc)
    avg_dist_good.append(newDistance)
    improvements.append(round(100 * (oldDistance - newDistance) / oldDistance, 1))
    print(f"Improvement: {100 * (oldDistance - newDistance) / oldDistance}%")
print(improvements)
# print(avg_dist_bad)
# print(avg_dist_good)
print(f"Overall Avg Improvement: {sum(improvements) / len(improvements)}%")
