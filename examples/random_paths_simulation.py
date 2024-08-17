########### RUN RANDOM PATHS SIMULTAION #################
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.curdir))

from src.gridload import bfs_path, load_grid_graph, plot_grid


grid, dock_map = load_grid_graph("Crossdock Maps/DC820.csv")
pickup_deliveries = [
    [random.choice(list(dock_map.keys())), random.choice(list(dock_map.keys()))]
    for i in range(50)
]
pickup_deliveries += [[random.choice(list(dock_map.keys())), "IBNP"] for i in range(5)]
pickup_deliveries += [["IBNP", random.choice(list(dock_map.keys()))] for i in range(5)]
pickup_deliveries += [["MNOB", random.choice(list(dock_map.keys()))] for i in range(5)]
pickup_deliveries += [[random.choice(list(dock_map.keys())), "MNOB"] for i in range(5)]
pickup_deliveries += [["621", random.choice(list(dock_map.keys()))] for i in range(5)]
pickup_deliveries += [[random.choice(list(dock_map.keys())), "621"] for i in range(5)]
pickup_deliveries += [["631", random.choice(list(dock_map.keys()))] for i in range(5)]
pickup_deliveries += [[random.choice(list(dock_map.keys())), "631"] for i in range(5)]
random.shuffle(pickup_deliveries)

paths = []
for pair in pickup_deliveries:
    if pair[0] == pair[1]:
        continue
    path = bfs_path(grid, dock_map[str(pair[0])], dock_map[str(pair[1])])
    paths.append(path)
plot_grid(grid, paths)
