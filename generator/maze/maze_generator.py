from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators


def generate_dataset(n_mazes, grid_n, directory):
    # generate dataset according to parameters
    # generate around 100_000? mazes
    return


def main():
    cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="train", # name is only for you to keep track of things
        grid_n=5, # number of rows/columns in the lattice
        n_mazes=4, # number of mazes to generate
        maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
        maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
    )

    dataset: MazeDataset = MazeDataset.from_config(cfg)

    print(dataset[0].as_ascii())

if __name__ == '__main__':
    main()