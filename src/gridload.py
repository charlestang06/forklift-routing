"""
Created by: Charles Tang
For:        BJ's Wholesale Robotics Team
Summary:    CSV parsing utilities for Forklift VRP for Crossdeck Map and Route Generation (BFS)
"""

############## IMPORTS #################
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation


############# GRID UTILITIES ##################
def load_grid_graph(filename):
    """
    Summary: Loads a grid graph from a CSV file into a 2D list.
    Inputs: (str) filename (path to CSV).
    Outputs: A 2D list representing the grid graph.
    """
    dock_map = {}  # map between dock and coord in graph
    grid = []
    # read in CSV map
    with open(filename, "r", encoding="utf-8-sig") as file:
        line_number = 0
        for line in file:
            row = line.strip().split(",")
            row = [str(cell).strip() for cell in row]
            for i in range(len(row)):
                # X = Wall, R = cannot go left, L = cannot go right
                if len(row[i]) != 0 and row[i] not in ["X", "R", "L"]:
                    dock_map[str(row[i])] = (line_number, i)
                    row[i] = ""
            grid.append(row)
            line_number += 1
    return grid, dock_map


def plot_grid(grid, paths=None, bubbles=None):
    """
    Summary: Plots the grid represented by the 2D list on a Matplotlib figure.
    Inputs: grid: A 2D list representing the grid graph.
    Outputs: Plotted grid, with animated paths OR bubbles on hotspots
    """
    rows, cols = len(grid), len(grid[0])
    forklifts = 1

    cmap = {
        "X": "black",  # Wall color
        "": "white",  # Empty space color (and all other elements)
        "R": "white",
        "L": "white",
        "P": [
            "green",
            "red",
            "pink",
            "blue",
            "grey",
            "purple",
            "orange",
            "yellow",
        ],  # Path color (random)
    }
    # Create a new figure and axes
    plt.rcParams["animation.ffmpeg_path"] = "C:/ffmpeg/bin/ffmpeg.exe"
    fig, ax = plt.subplots()

    # Plot map as a grid of cells
    for row in range(rows):
        for col in range(cols):
            cell_value = grid[row][col]
            cell_color = cmap[cell_value]
            ax.add_patch(Rectangle((col, row), 1, 1, color=cell_color))
    # Plot bubbles based on count
    for coord, count in bubbles:
        row, col = coord
        bubble_size = (
            count**0.5 * 0.05  # **0.8 * 0.005
        )  # Adjust multiplier for bubble size based on count

        # Create and add a circle patch for the bubble
        circle = Circle(
            (col + 0.5, row + 0.5),
            bubble_size,
            color="#D31242",
            alpha=0.75,
        )
        ax.add_patch(circle)

    # Make cell size smaller for printing paths
    patch_size = 0.25

    # Queue in order to remove cells from paths after some time has passed
    delete_queue = []

    def animate(i):
        """
        Callback function for animation loop
        """
        if not paths:
            return
        # loop through the number of paths able to be made by forklifts
        for path in paths[: min(forklifts, len(paths))]:
            if not path or len(path) == 0:
                paths.remove(path)
                continue
            path_color = cmap["P"][
                path[-1][1] % len(cmap["P"])
            ]  # want path_color to stay constant by using last value of path
            row, col = path[0]
            # put colored cell where path is
            ax.add_patch(
                Rectangle(
                    (col + 0.5 - patch_size / 2, row + 0.5 - patch_size / 2),
                    patch_size,
                    patch_size,
                    color=path_color,
                    alpha=0.5,
                )
            )
            # add cell to queue for deletion later
            delete_queue.append(path.pop(0))
        # delete cells from deletion queue
        if i > 4 * forklifts:
            for x in range(forklifts):
                if len(delete_queue) == 0:
                    continue
                row, col = delete_queue.pop(0)
                ax.add_patch(
                    Rectangle(
                        (col + 0.5 - patch_size / 2, row + 0.5 - patch_size / 2),
                        patch_size,
                        patch_size,
                        color="white",
                        alpha=1,
                    )
                )

    # Set axis limits and labels
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Remove ticks and grid lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Display the plot
    animate(0)
    # anim = animation.FuncAnimation(fig, animate, frames=len(paths) * 200, interval=500)

    # Write to mp4
    anim = animation.FuncAnimation(fig, animate, frames=len(paths) * 200)  # type: ignore
    ff_writer = animation.FFMpegWriter(fps=10)
    anim.save("animation.mp4", writer=ff_writer)
    plt.show()


def bfs_path(grid, start, end):
    """
    Summary: Calculates the shortest distance between two
             dock coordinates in a grid graph using BFS.

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
        parent = {}
        while queue:
            row, col, distance = queue.pop(0)
            if (row, col) == end:
                path = []
                # Reconstruct the path by following parent nodes
                while (row, col) in parent:
                    path.append((row, col))
                    (row, col) = parent[(row, col)]
                path.reverse()  # Reverse to get the path in the correct order (start to end)
                return path  # Return the path coordinates

            if (row, col) not in visited:
                visited.add((row, col))

                # Explore adjacent cells (up, down, left, right)
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                if grid[row][col] == "R":
                    directions.remove((0, -1))
                if grid[row][col] == "L":
                    directions.remove((0, 1))
                random.shuffle(directions)
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if (
                        0 <= new_row < rows
                        and 0 <= new_col < cols
                        and grid[new_row][new_col] != "X"
                        and (new_row, new_col) not in visited
                    ):
                        queue.append((new_row, new_col, distance + 1))
                        # FIX PARENT ALGORITHM SO IT IS MORE NATURAL
                        parent[(new_row, new_col)] = (row, col)
