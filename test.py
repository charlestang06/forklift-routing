"""
Functions for loading in historical data from test data and backtesting models on the data
Created by: Charles Tang
For: BJ's Wholesale Robotics Team
"""

from distance_map import load_distance_map
from models import RoutingModel


def backtest_model(model: RoutingModel, dc: int) -> float:
    """
    Loads in RoutingModel and backtests on a historical tasks
    Inputs: model - RoutingModel
    Outputs: avg_distance - average distance per pallet averaged across all Xdockers
    """
    avg_distances = []
    total_distances = []
    total_pallets = 0

    # Allocate routes
    routes = model.assign_tasks()

    # Load dist_map if not provided
    dist_map = model.distance_map
    if not dist_map:
        dist_map = load_distance_map(dc)

    # Loop through each user
    for user in routes.keys():
        dist = 0
        route = routes[user]
        num_pallets = len(route)
        total_pallets += num_pallets
        # Loop through user's route
        for i in range(len(route)):
            # task distance
            dist += dist_map.get((route[i][0], route[i][1]), 30)
            if i < len(route) - 1:
                # distance to next task
                dist += dist_map.get((route[i][1], route[i + 1][0]), 30)
        # Add to data
        if num_pallets > 0:
            avg_distances.append(round(dist / num_pallets, 0))
            total_distances.append(dist)

    # Average of total distances
    return round(sum(total_distances) / total_pallets * 6.67, 0)
