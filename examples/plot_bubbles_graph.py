########### EXAMPLE CODE FOR BUBBLES PLOTTING ##############
import sys, os

sys.path.append(os.path.abspath(os.path.curdir))

from src.data_load import get_shipping_doors_volume, load_aggregate_data
from src.distance_map import load_distance_map
from src.gridload import load_grid_graph, plot_grid

# Define parameters
DC = 820

# Load distance map and grid
old_distance_map = load_distance_map(dc=DC)
grid, dock_map = load_grid_graph(f"Crossdock Maps/DC{DC}.csv")

# Get volume per door
bubbles = []
shipping_door_volume = sorted(
    get_shipping_doors_volume(tasks=load_aggregate_data(dc=DC), dc=DC),
    key=lambda x: x[1],
    reverse=True,
)

# Remap bubbles based on mapping from ShippingLaneModel
old_new_door_map = {
    # three time rotation
    58: 106,
    106: 110,
    110: 58,
    #
    82: 82,
    #
    165: 84,
    84: 165,
    #
    138: 86,
    86: 138,
    #
    52: 88,
    88: 52,
    #
    172: 108,
    108: 172,
    #
    36: 112,
    112: 36,
    #
    66: 114,
    114: 66,
    #
    28: 116,
    116: 28,
}

# Map bubbles with volume
bubbles = []
print(shipping_door_volume[:10])
for x in sorted(shipping_door_volume, key=lambda x: x[1], reverse=True):
    if x[0] in old_new_door_map:
        print()
        bubbles.append((dock_map[str(old_new_door_map[int(x[0])])], x[1]))
# Plot bubbles
TOP = 5
plot_grid(grid, bubbles=bubbles[:TOP])
