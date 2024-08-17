# Calculate distance_map given a csv defined distribution center

import sys, os
import json

sys.path.append(os.path.abspath(os.path.curdir))

from src.distance_map import calculate_distance_map
from src.gridload import load_grid_graph


DC = 800  # Change DC number
grid, dock_map = load_grid_graph(f"Crossdock Maps/DC{DC}.csv")

# Calculate new distance map. Note this may take some time.
print("This may take some time...")
distance_map = calculate_distance_map(dock_map, grid)
string_keys_data = {str(k): distance_map[k] for k in distance_map}

# # Dump the dictionary with string keys to JSON
with open(f"Crossdock Maps/DC{DC}.json", "w", encoding="utf-8") as outfile:
    json.dump(string_keys_data, outfile)
