"""
Created by: Charles Tang
For:        BJ's Wholesale Robotics Team
Summary:    Distance utilities for Forklift VRP 
"""

import json
from typing import List


def find_distance(
    dock1: int, dock2: int, dock_map: dict, grid: List[List], method="bfs"
) -> int:
    """
    Summary: Find distance between two docks
    Inputs:  (int) dock 1 number
             (int) dock 2 number
             (dict) dock # -> (list) 2d coordinate
             (List[List]) grid
    Outputs: (int) Manhattan distance between two docks
    """
    try:
        dock1_coord = dock_map[str(dock1)]
    except KeyError:
        print(f"Dock 1 ({dock1}) does not exist!")
    try:
        dock2_coord = dock_map[str(dock2)]
    except KeyError:
        print(f"Dock 2 ({dock2}) does not exist!")

    # Manhattan distance abs(x1-x2) + abs(y1-y2)
    # replace w A-star or some other fancy algo if needed
    if method == "bfs":
        return bfs_distance(grid, dock1_coord, dock2_coord)
    return abs(dock1_coord[0] - dock2_coord[0]) + abs(dock1_coord[1] - dock2_coord[1])


def bfs_distance(grid: List[List], start: tuple, end: tuple) -> int:
    """
    Summary: Calculates the shortest distance between two dock
             coordinates in a grid graph using BFS.

    Inputs:
        grid: A 2D list representing the grid graph, where 'X' represents walls.
        start: A tuple (row, col) representing the starting coordinates.
        end: A tuple (row, col) representing the ending coordinates.

    Outputs:
        The shortest distance between start and end coordinates, or -1 if no path exists.
    """

    rows, cols = len(grid), len(grid[0])

    # Check if start and end coordinates are within grid bounds
    if (
        0 <= start[0] < rows
        and 0 <= start[1] < cols
        and 0 <= end[0] < rows
        and 0 <= end[1] < cols
    ):
        # Visited set to track explored cells
        visited = set()
        queue = [(start[0], start[1], 0)]  # (row, col, distance)

        while queue:
            row, col, distance = queue.pop(0)
            if (row, col) == end:
                return distance

            if (row, col) not in visited:
                visited.add((row, col))

                # Explore adjacent cells (up, down, left, right)
                # no need to understand highway directions because it is unintuitive
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_row, new_col = row + dr, col + dc
                    if (
                        0 <= new_row < rows
                        and 0 <= new_col < cols
                        and grid[new_row][new_col] != "X"
                        and (new_row, new_col) not in visited
                    ):
                        queue.append((new_row, new_col, distance + 1))

    return -1  # No path found


def calculate_distance_map(dock_map: dict, grid: List[List]) -> dict:
    """
    Summary: Find all distance pairs between two docks
    Inputs:  (dict) dock_map where dock # -> (list) 2d coordinate
    Outputs: (dict) (dock1, dock2) -> (int) distance between
    """
    distance_map = {}
    for dock1 in dock_map.keys():
        for dock2 in dock_map.keys():
            if dock1 == dock2:
                continue
            distance = find_distance(dock1, dock2, dock_map, grid, method="bfs")
            distance_map[(dock1, dock2)] = distance
            distance_map[(dock2, dock1)] = distance
    return distance_map


def load_distance_map(dc: int) -> dict:
    """
    Summary: Load distance map from JSON file
    Inputs:  (int) DC number
    Outputs: (dict) (dock1, dock2) -> (int) distance between
    """
    distance_map = {}
    with open(f"Crossdock Maps/DC{dc}.json", "r", encoding="utf-8") as infile:
        distance_map = json.load(infile)
    new_dist_map = {}
    for x in distance_map:
        values = x[1:-1].split(",")
        new_tuple = (values[0].strip()[1:-1], values[1].strip()[1:-1])
        new_dist_map[new_tuple] = distance_map[x]
    return new_dist_map
