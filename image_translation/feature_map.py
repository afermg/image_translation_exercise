"""
Script to visualize the encoder feature maps of a trained model.
Using PCA to visualize feature maps is inspired by
https://doi.org/10.48550/arXiv.2304.07193 (Oquab et al., 2023).
"""

from typing import NamedTuple

import numpy as np
import torch
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA


# Defining the colors for plotting
class Color(NamedTuple):
    r: float
    g: float
    b: float


def feature_map_pca(feature_map: np.array, n_components: int = 8) -> PCA:
    """
    Compute PCA on a feature map.
    :param np.array feature_map: (C, H, W) feature map
    :param int n_components: number of components to keep
    :return: PCA: fit sklearn PCA object
    """
    # (C, H, W) -> (C, H*W)
    feat = feature_map.reshape(feature_map.shape[0], -1)
    pca = PCA(n_components=n_components)
    pca.fit(feat)
    return pca


def pcs_to_rgb(feat: np.ndarray, n_components: int = 8) -> np.ndarray:
    pca = feature_map_pca(feat[0], n_components=n_components)
    pc_first_3 = pca.components_[:3].reshape(3, *feat.shape[-2:])
    return np.stack(
        [rescale_intensity(pc, out_range=(0, 1)) for pc in pc_first_3], axis=-1
    )


# Defining the functions to rescale the image and composite the nuclear and membrane images
def rescale_clip(image: torch.Tensor) -> np.ndarray:
    return rescale_intensity(image, out_range=(0, 1))[..., None].repeat(3, axis=-1)


def composite_nuc_mem(
    image: torch.Tensor, nuc_color: Color, mem_color: Color
) -> np.ndarray:
    c_nuc = rescale_clip(image[0]) * nuc_color
    c_mem = rescale_clip(image[1]) * mem_color
    return rescale_intensity(c_nuc + c_mem, out_range=(0, 1))


def clip_p(image: np.ndarray) -> np.ndarray:
    return rescale_intensity(image.clip(*np.percentile(image, [1, 99])))
