import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np


class ReflectanceDataset(Dataset):
    def __init__(
        self,
        positions: np.ndarray,  # (N, 3) array of 3D positions
        view_dirs: np.ndarray,  # (N, 3) array of view directions
        light_dirs: np.ndarray,  # (N, 3) array of light directions
        colors: np.ndarray,  # (N, 3) array of RGB colors
    ):
        self.positions = torch.FloatTensor(positions)
        self.view_dirs = torch.FloatTensor(view_dirs)
        self.light_dirs = torch.FloatTensor(light_dirs)
        self.colors = torch.FloatTensor(colors)

        self.train_data, self.val_data = self._prepare_training_data()

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return {
            "position": self.positions[idx],
            "view_dir": self.view_dirs[idx],
            "light_dir": self.light_dirs[idx],
            "color": self.colors[idx],
        }

    def _prepare_training_data(self) -> Tuple[Tuple, Tuple]:
        num_samples = self.positions.shape[0]
        split_index = int(0.8 * num_samples)

        train_data = (
            self.positions[:split_index],
            self.view_dirs[:split_index],
            self.light_dirs[:split_index],
            self.colors[:split_index],
        )
        val_data = (
            self.positions[split_index:],
            self.view_dirs[split_index:],
            self.light_dirs[split_index:],
            self.colors[split_index:],
        )

        return train_data, val_data
