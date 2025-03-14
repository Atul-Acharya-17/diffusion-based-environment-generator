import os
import random
import numpy as np
import torch

from utils.maze_dataset import MazeTensorDataset

def preprocess_image(image, target_size=32):
    image = np.array(image)
    
    scale_factor = target_size // image.shape[0] 
    image = np.kron(image, np.ones((scale_factor, scale_factor, 1))) 
    
    image = image.astype(np.float32) / 127.5 - 1
    image = torch.tensor(image).permute(2, 0, 1)
    return image

def load_dataset_from_npy(directory="./data", target_size=32):
    images = []
    path_lengths = []
    
    files = sorted([f for f in os.listdir(directory) if f.endswith(".npy")])
    
    for file in files:
        img = np.load(os.path.join(directory, file))
        
        mask = np.all(img == [0, 0, 255], axis=-1)
        img[mask] = [255, 255, 255]
        img = img[:-1, :-1] 
        
        image = preprocess_image(img, target_size)
        
        base_name = os.path.splitext(file)[0]
        len_filename = base_name + "_len.txt"
        len_path = os.path.join(directory, len_filename)
        
        with open(len_path, "r") as f:
            maze_length = int(f.read().strip())
        
        images.append(image)
        path_lengths.append(maze_length)
    
    return images, path_lengths

def get_train_test_dataset(images: list, path_lengths: list) -> tuple[MazeTensorDataset, list, MazeTensorDataset, list]:
    total = len(images)
    test_size = int(0.2 * total)
    all_indices = list(range(total))
    random.shuffle(all_indices)

    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]

    train_images = [images[i] for i in train_indices]
    train_path_lengths = [path_lengths[i] for i in train_indices]

    test_images = [images[i] for i in test_indices]
    test_path_lengths = [path_lengths[i] for i in test_indices]

    dataset = MazeTensorDataset(train_images, train_path_lengths)
    test_dataset = MazeTensorDataset(test_images, test_path_lengths)

    return dataset, train_path_lengths, test_dataset, test_path_lengths

