import os
import numpy as np
from heapq import heappop, heappush
from tqdm import tqdm

def load_maze(file_path):
    return np.load(file_path)

def get_coordinates(maze):
    source = np.argwhere(maze[:, :, 1] == 1)
    destination = np.argwhere(maze[:, :, 2] == 1)

    if len(source) == 0 or len(destination) == 0:
        return None, None

    return tuple(source[0]), tuple(destination[0])

def a_star(maze, distance_fn: Callable[[Tuple[int, int], Tuple[int, int]], int]):
    height, width, _ = maze.shape
    start, end = get_coordinates(maze)

    if start is None or end is None:
        return 0

    open_set = [(0, start)]
    g_scores = {start: 0}
    visited = set()
    nodes_explored = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        _, (y, x) = heappop(open_set)
        nodes_explored += 1
        visited.add((y, x))

        if (y, x) == end:
            return nodes_explored - 2

        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            if 0 <= ny < height and 0 <= nx < width and maze[ny, nx, 0] == 1:
                new_g = g_scores[(y, x)] + 1
                if (ny, nx) not in g_scores or new_g < g_scores[(ny, nx)]:
                    g_scores[(ny, nx)] = new_g
                    f_score = new_g + distance_fn((ny, nx), end)
                    heappush(open_set, (f_score, (ny, nx)))

    return nodes_explored - 2

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def main(parent_directory):
    input_directory = f'{parent_directory}/mazes/'
    output_directory = f"{parent_directory}/a_star_l2_results/"

    os.makedirs(output_directory, exist_ok=True)

    maze_files = [f for f in os.listdir(input_directory) if f.endswith(".npy") and "maze" in f]

    with tqdm(total=len(maze_files), desc="Processing Mazes", unit="maze") as pbar:
        for file in maze_files:
            idx = int(file.split("_")[1].split(".")[0])
            file_path = os.path.join(input_directory, file)

            maze = np.load(file_path)
            nodes_explored = a_star(maze, euclidean_distance)

            output_file = os.path.join(output_directory, f"a_star_{idx}.npy")
            np.save(output_file, np.array(nodes_explored, dtype=np.int32))

            output_file = os.path.join(output_directory, f"a_star_{idx}.txt")
            with open(output_file, "w") as f:
                f.write(str(nodes_explored))

            pbar.update(1)

if __name__ == '__main__':
    main("./data")