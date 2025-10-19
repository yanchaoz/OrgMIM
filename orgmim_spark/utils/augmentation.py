import cv2
import time
import math
import random
import torch
import numpy as np
from scipy.ndimage import zoom


def produce_simple_aug(data: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """
    Apply simple flipping and transposing augmentation on 3D data.

    Args:
        data: 3D numpy array (z, y, x).
        rule: Binary array of length 4, controlling the flips and transpose.
    """
    assert data.ndim == 3 and len(rule) == 4
    if rule[0]:
        data = data[::-1, :, :]
    if rule[1]:
        data = data[:, :, ::-1]
    if rule[2]:
        data = data[:, ::-1, :]
    if rule[3]:
        data = data.transpose(0, 2, 1)
    return data


class SimpleAugment(object):
    """
    Routine 3D data augmentation, including random flips and transposition.
    There are 2^4 = 16 possible transformations.
    """

    def __init__(self, skip_ratio: float = 0.5):
        self.skip_ratio = skip_ratio

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return self.forward(inputs)

    def forward(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if np.random.rand() < self.skip_ratio:
            rule = np.random.randint(2, size=4)
            for idx in range(len(inputs)):
                inputs[idx] = produce_simple_aug(inputs[idx], rule)
        return inputs


class RandomIntensity(object):
    """
    Randomly adjust contrast and brightness for 3D image data.
    """

    def __init__(self,
                 skip_ratio: float = 0.5,
                 contrast_factor: float = 0.1,
                 brightness_factor: float = 0.1):
        self.skip_ratio = skip_ratio
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.skip_ratio:
            imgs = self._augment3d(inputs)
            imgs = np.clip(imgs, 0, 1)
            return imgs
        else:
            return inputs

    def _augment3d(self, imgs: np.ndarray) -> np.ndarray:
        imgs = imgs * (1 + (np.random.rand() - 0.5) * self.contrast_factor)
        imgs = imgs + (np.random.rand() - 0.5) * self.brightness_factor
        imgs = np.clip(imgs, 0, 1)
        # Random gamma adjustment
        gamma = 2.0 ** (np.random.rand() * 2 - 1)
        imgs = imgs ** gamma
        return imgs


class RandomScaling(object):
    """
    Random 3D scaling augmentation for volumetric data.
    """

    def __init__(self,
                 scale_range: tuple[float, float] = (0.9, 1.1),
                 skip_ratio: float = 0.5,
                 order: int = 1):
        """
        Args:
            scale_range: (min_scale, max_scale), scaling factor range.
            skip_ratio: Probability of performing scaling.
            order: Interpolation order (0=nearest, 1=linear, 3=cubic).
        """
        self.scale_range = scale_range
        self.skip_ratio = skip_ratio
        self.order = order

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return self.forward(inputs)

    def forward(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if np.random.rand() < self.skip_ratio:
            scale_factor = np.random.uniform(*self.scale_range)
            scaled_inputs = []
            for arr in inputs:
                scaled_inputs.append(zoom(arr, zoom=scale_factor, order=self.order))
            return scaled_inputs
        else:
            return inputs
