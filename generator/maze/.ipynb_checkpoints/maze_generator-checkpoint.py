import argparse
import numpy as np
import os
import random
from tqdm import tqdm

from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.generation import LatticeMazeGenerators


def get_required_maze(dataset: MazeDataset, path_length: int):
    return list(filter(lambda maze: len(maze.solution) == path_length, dataset))

def generate_dataset(n_mazes, directory):
    maze_dim = 8
    
    algorithms = {
        "gen_dfs": LatticeMazeGenerators.gen_dfs,
        "gen_wilson": LatticeMazeGenerators.gen_wilson,
        "gen_prim": LatticeMazeGenerators.gen_prim
    }

    idx = 0
    # min_path_length = 2
    # max_path_length = 40
    # path_length = min_path_length
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
                n_mazes=1000, # number of mazes to generate
                maze_ctor=algorithm # algorithm to generate the maze
                
            )
            
            dataset: MazeDataset = MazeDataset.from_config(cfg)

            print("generated")
            # required_mazes = get_required_maze(dataset, path_length)
            # print(len(required_mazes))

            for maze in dataset:
                maze_pixels = maze.as_pixels()  # Shape: (height, width, 3)

                maze_filename = os.path.join(directory, f"maze_{idx}.npy")
                np.save(maze_filename, maze_pixels)

                length_filename = os.path.join(directory, f"maze_{idx}_len.txt")
                with open(length_filename, "w") as f:
                    f.write(str(len(maze.solution)))

                idx += 1

            # path_length = (path_length + 1) % (max_path_length + 1)
            # if path_length < min_path_length:
            #     path_length = min_path_length

            pbar.update(len(dataset))