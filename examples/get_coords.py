# Get coordinate locations of each of the dock doors

import sys, os
import json

sys.path.append(os.path.abspath(os.path.curdir))

from src.gridload import load_grid_graph

DC = 840
grid, dock_map = load_grid_graph(filename=f"Crossdock Maps/DC{DC}.csv")

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(dock_map, f)
