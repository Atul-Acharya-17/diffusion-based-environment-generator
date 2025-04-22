import random
from collections import deque
from tqdm import tqdm
import numpy as np
import os

def generate_grid_world(size=10):    
    num_walls = random.randint(0, 90)
    
    grid = [[1 for _ in range(size)] for _ in range(size)]
    
    wall_positions = set()
    while len(wall_positions) < num_walls:
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        grid[x][y] = 0
        wall_positions.add((x, y))
    
    while True:
        start = (random.randint(0, size-1), random.randint(0, size-1))
        end = (random.randint(0, size-1), random.randint(0, size-1))
        if start != end and grid[start[0]][start[1]] == 1 and grid[end[0]][end[1]] == 1:
            break

    return grid, start, end

def bfs(grid, start, end):
    size = len(grid)
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == end:
            return dist
        
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited and grid[nx][ny] == 1:
                queue.append((nx, ny, dist + 1))
    
    return None

def generate_multiple_grid_worlds(num_worlds, size=10):
    maze_count = 0
    directory = "./data/"

    os.makedirs(directory, exist_ok=True)
    
    with tqdm(total=num_worlds, desc="Generating Grid Worlds") as pbar:
        while maze_count < num_worlds:
            grid, start, end = generate_grid_world(size)
            path_length = bfs(grid, start, end)

            if path_length is not None:
                grid_world = np.zeros((size, size, 3), dtype=np.uint8)
                grid_world[:, :, 0] = grid
                grid_world[start[0], start[1], 1] = 1
                grid_world[end[0], end[1], 2] = 1

                maze_filename = os.path.join(directory, f"maze_{maze_count}.npy")
                path_length_filename = os.path.join(directory, f"path_length_{maze_count}.npy")
                np.save(maze_filename, grid_world)
                np.save(path_length_filename, path_length)

                maze_count += 1
                pbar.update(1)

def main():
    generate_multiple_grid_worlds(num_worlds=100_000)

if __name__ == '__main__':
    main()