import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


def make_point_clouds(n_samples_per_shape: int, n_points: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate point clouds for circles, spheres, and tori with random noise.

    Returns:
        point_clouds: array of shape (3 * n_samples_per_shape, n_points**2, 3) for sphere/torus,
                      and (n_points**2, 3) for circle (kept consistent here by using n_points**2)
        labels: array of shape (3 * n_samples_per_shape,)
    """
    circle_point_clouds = [
        np.asarray(
            [
                [
                    np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    0,
                ]
                for t in range((n_points ** 2))
            ]
        )
        for _ in range(n_samples_per_shape)
    ]
    circle_labels = np.zeros(n_samples_per_shape)

    sphere_point_clouds = [
        np.asarray(
            [
                [
                    np.cos(s) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.cos(s) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for _ in range(n_samples_per_shape)
    ]
    sphere_labels = np.ones(n_samples_per_shape)

    torus_point_clouds = [
        np.asarray(
            [
                [
                    (2 + np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (2 + np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for _ in range(n_samples_per_shape)
    ]
    torus_labels = 2 * np.ones(n_samples_per_shape)

    point_clouds = np.concatenate((circle_point_clouds, sphere_point_clouds, torus_point_clouds))
    labels = np.concatenate((circle_labels, sphere_labels, torus_labels))
    return point_clouds, labels


class GiottoPointCloudDataset(Dataset):
    def __init__(self, point_clouds: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(point_clouds, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


