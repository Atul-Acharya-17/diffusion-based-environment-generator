import argparse
import numpy as np
import os
import random
from tqdm import tqdm

from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.generation import LatticeMazeGenerators


def get_required_maze(dataset: MazeDataset, path_length: int):
    return list(filter(lambda maze: len(maze.solution) == path_length, dataset))

def ascii_to_numpy(maze_ascii):
    maze_2d = np.array([list(row) for row in maze_ascii.strip().split("\n")])

    if 'S' not in maze_ascii:
        print(maze_ascii)

    if 'E' not in maze_ascii:
        print(maze_ascii)

    base = (maze_2d != "#").astype(np.uint8)
    source = (maze_2d == "S").astype(np.uint8)
    destination = (maze_2d == "E").astype(np.uint8)

    return np.dstack((base, source, destination))

def generate_dataset(n_mazes, directory):
    maze_dim = 5
    
    algorithms = {
        "gen_dfs": LatticeMazeGenerators.gen_dfs
        # "gen_wilson": LatticeMazeGenerators.gen_wilson
        # "gen_prim": LatticeMazeGenerators.gen_prim
    }

    idx = 0
    min_path_length = 2
    max_path_length = 70
    path_length = min_path_length
    seed = 0

    os.makedirs(directory, exist_ok=True)

    with tqdm(total=n_mazes, desc="Generating Mazes", unit="maze") as pbar:
        while idx < n_mazes:
            seed += 1
            
            algorithm_name, algorithm = random.choice(list(algorithms.items()))

            cfg: MazeDatasetConfig = MazeDatasetConfig(
                seed=seed,
                name="train", # name is only for you to keep track of things
                grid_n=maze_dim, # number of rows/columns in the lattice
                n_mazes=10000, # number of mazes to generate
                maze_ctor=algorithm, # algorithm to generate the maze
                maze_ctor_kwargs=dict(do_forks=True), 
            )

            dataset: MazeDataset = MazeDataset.from_config(cfg)

            required_mazes = get_required_maze(dataset, path_length)

            print(len(required_mazes))

            for maze in required_mazes:

                maze_numpy = ascii_to_numpy(maze.as_ascii())
                maze_path_length = maze.as_ascii().count("X")

                maze_filename = os.path.join(directory, f"maze_{idx}.npy")
                path_length_filename = os.path.join(directory, f"path_length_{idx}.npy")
                np.save(maze_filename, maze_numpy)
                np.save(path_length_filename, maze_path_length)

                idx += 1

            path_length = (path_length + 1) % (max_path_length + 1)
            if path_length < min_path_length:
                path_length = min_path_length

            pbar.update(len(required_mazes))


def main():
    # TODO: make this configurable  
    generate_dataset(100_000, "./data/")

if __name__ == '__main__':
    main()