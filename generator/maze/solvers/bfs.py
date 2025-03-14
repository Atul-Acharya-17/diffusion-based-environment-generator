import numpy as np
import os
from collections import deque
from tqdm import tqdm

def load_maze(file_path):
    return np.load(file_path)

def numpy_to_ascii(maze_numpy):
    """Converts a (H, W, 3) NumPy array back to an ASCII maze string."""
    height, width, _ = maze_numpy.shape
    ascii_maze = np.full((height, width), " ", dtype="<U1")  # Default to paths (" ")

    ascii_maze[maze_numpy[:, :, 0] == 0] = "#"

    source_pos = np.argwhere(maze_numpy[:, :, 1] == 1)
    for y, x in source_pos:
        ascii_maze[y, x] = "S"

    dest_pos = np.argwhere(maze_numpy[:, :, 2] == 1)
    for y, x in dest_pos:
        ascii_maze[y, x] = "E"

    return "\n".join("".join(row) for row in ascii_maze)  # Convert array to ASCII string

def get_coordinates(maze):
    source = np.argwhere(maze[:, :, 1] == 1)  # 2nd channel == 1 -> Source
    destination = np.argwhere(maze[:, :, 2] == 1)  # 3rd channel == 1 -> Destination

    if len(source) == 0 or len(destination) == 0:
        print("Error Maze")
        print(maze)
        print(numpy_to_ascii(maze))
        return None, None

    return tuple(source[0]), tuple(destination[0])

def bfs_shortest_path(maze):
    """Finds the shortest path using BFS. Returns path length or -1 if unreachable."""
    height, width, _ = maze.shape
    start, end = get_coordinates(maze)

    if start is None or end is None:
        return 0

    queue = deque([(start[0], start[1], 0)])
    visited = set()
    nodes_explored = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x, dist = queue.popleft()
        nodes_explored += 1

        if (y, x) == end:
            return dist

        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            if 0 <= ny < height and 0 <= nx < width and maze[ny, nx, 0] == 1 and (ny, nx) not in visited:
                queue.append((ny, nx, dist + 1))
                visited.add((ny, nx))

    return nodes_explored - 2 # exclude start and end

def main():
    # TODO: Make this configurable
    input_directory = './data/'
    output_directory = "./data/bfs_results/"

    os.makedirs(output_directory, exist_ok=True)

    maze_files = [f for f in os.listdir(input_directory) if f.endswith(".npy") and "maze" in f]

    with tqdm(total=len(maze_files), desc="Processing Mazes", unit="maze") as pbar:
        for file in maze_files:
            idx = int(file.split("_")[1].split(".")[0])
            file_path = os.path.join(input_directory, file)

            maze = np.load(file_path)
            nodes_explored = bfs_shortest_path(maze)

            output_file = os.path.join(output_directory, f"bfs_{idx}.npy")
            np.save(output_file, np.array(nodes_explored, dtype=np.int32))

            output_file = os.path.join(output_directory, f"bfs_{idx}.txt")
            with open(output_file, "w") as f:
                f.write(str(nodes_explored))

            pbar.update(1)


if __name__ == '__main__':
    main()