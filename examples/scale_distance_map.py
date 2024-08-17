# Scale distances to feet
# multiply all distances in jsons by 6 and write to another json

import sys, os
import json

sys.path.append(os.path.abspath(os.path.curdir))

DC = 840
with open(f"Crossdock Maps/DC{DC}.json", mode="r", encoding="utf-8") as f:
    new_dist_map = {}
    dist_map = json.load(f)
    for key, value in dist_map.items():
        new_dist_map[key] = value * 6

with open("data.json", "w", encoding="utf-8") as f2:
    json.dump(new_dist_map, f2)
