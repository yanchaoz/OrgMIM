import imageio
import cv2
import numpy as np
from scipy.ndimage import shift
from sklearn.decomposition import PCA
import sys
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import zoom
import os

# Add the project root directory to the system path
sys.path.append("..")


def nearest_neighbor_resize(image, new_size):
    """
    Resize an image using nearest-neighbor interpolation.

    Args:
        image (np.ndarray): Input image.
        new_size (tuple): Target size (height, width).

    Returns:
        np.ndarray: Resized image.
    """
    old_height, old_width = image.shape
    new_height, new_width = new_size

    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            old_i = int(i * old_height / new_height)
            old_j = int(j * old_width / new_width)
            resized_image[i, j] = image[old_i, old_j]

    return resized_image


def embedding_pca(embeddings, n_components=3, as_rgb=True):
    """
    Perform PCA dimensionality reduction on embeddings and optionally convert to RGB format.

    Args:
        embeddings (np.ndarray): Input embeddings.
        n_components (int): Number of dimensions after PCA reduction.
        as_rgb (bool): Whether to convert the result to RGB format.

    Returns:
        np.ndarray: Reduced embeddings.
    """
    if as_rgb and n_components != 3:
        raise ValueError("n_components must be 3 when as_rgb=True.")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')

    return embed_flat


def _embeddings_to_probabilities(embed1, embed2, delta_v, delta_d, embedding_axis):
    """
    Compute probabilities between two embeddings.

    Args:
        embed1 (np.ndarray): First embedding.
        embed2 (np.ndarray): Second embedding.
        delta_v (float): Threshold value.
        delta_d (float): Scaling factor.
        embedding_axis (int): Axis of the embeddings.

    Returns:
        np.ndarray: Probability matrix.
    """
    dis = np.linalg.norm(embed1 - embed2, axis=embedding_axis)
    dis[dis <= delta_v] = 0
    probs = (2 * delta_d - dis) / (2 * delta_d)
    probs = np.maximum(probs, 0) ** 2
    return probs


def embeddings_to_affinities(embeddings, offsets=[[-1, 0], [0, -1]], delta_v=0.5, delta_d=1.5, invert=False):
    """
    Convert embeddings to affinity maps.

    Args:
        embeddings (np.ndarray): Input embeddings.
        offsets (list): List of offsets.
        delta_v (float): Threshold value.
        delta_d (float): Scaling factor.
        invert (bool): Whether to invert the result.

    Returns:
        np.ndarray: Affinity maps.
    """
    ndim = embeddings.ndim - 1
    if not all(len(off) == ndim for off in offsets):
        raise ValueError("Offset dimensions do not match embeddings.")

    n_channels = len(offsets)
    shape = embeddings.shape[1:]
    affinities = np.zeros((n_channels,) + shape, dtype='float32')

    for cid, off in enumerate(offsets):
        shift_off = [0] + [-o for o in off]
        shifted = shift(embeddings, shift_off, order=0, prefilter=False)
        affs = _embeddings_to_probabilities(embeddings, shifted, delta_v, delta_d, embedding_axis=0)
        affinities[cid] = affs

    if invert:
        affinities = 1. - affinities

    return affinities

