import os

def read_files(directory):
    maze_files = [f for f in os.listdir(directory) if f.endswith("_len.txt")]
    mazes = []
    for maze_file in maze_files:
        with open(os.path.join(directory, maze_file), "r") as f:
            path_length = int(f.read())
            mazes.append(path_length)

    return mazes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    mazes = read_files(args.directory)
    print(mazes)

