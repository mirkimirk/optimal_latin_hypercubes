"This module contains some subroutines for generating a Latin Hypercube sample."
import numpy as np


def scale_up_points(points, center, radius, target_n_points):
    """Scales existing Lhd points up to prepare them for merging.

    Parameters
    ----------
    points : np.ndarray
        Points to scale.
    center : np.ndarray
        Center of the current trust region.
    radius : float
        Radius of the current trust region.
    target_n_points : int
        Number of points.

    Returns
    -------
    scaled : np.ndarray
        Upscaled points.
    """
    lower = center - radius
    scaled = (points - lower) * target_n_points / (2 * radius)
    return scaled


def scale_down_points(points, center, radius, target_n_points):
    """Scales existing Lhd points down to make a Lhd.

    Parameters
    ----------
    points : np.ndarray
        Points to scale.
    center : np.ndarray
        Center of the current trust region.
    radius : float
        Radius of the current trust region.
    target_n_points : int
        Number of points.

    Returns
    -------
    scaled : np.ndarray
        Downscaled points.
    """
    lower = center - radius
    scaled = points / target_n_points * (2 * radius) + lower
    return scaled


def get_empty_bin_info(existing_upscaled, target_n_points):
    """Get info on empty bins.

    Parameters
    ----------
    existing_upscaled : np.ndarray
        Existing points carrying info on their bins
    target_n_points : int
        Number of points.

    Returns
    -------
    out : np.ndarray
        Array containing tuples specifying empty bins.
    """
    dim = existing_upscaled.shape[1]
    empty_bins = []
    all_bins = set(range(target_n_points))
    for j in range(dim):
        filled_bins = set(np.floor(existing_upscaled[:, j].astype(int)))
        empty_bins.append(sorted(all_bins - filled_bins))
    max_empty = max(map(len, empty_bins))

    out = np.full((max_empty, dim), -1)
    for j, empty in enumerate(empty_bins):
        out[: len(empty), j] = empty

    return out


def create_upscaled_lhs_sample(dim, n_samples, lhs_design="centered"):
    """Create random Lhs sample.

    Parameters
    ----------
    dim : int
        Number of variables in the design space.
    n_samples : int
        Number of total samples.
    lhs_design : str, optional
        One of "centered", "released". "Centered" places points in the middle of each
        bin and finds optimal midpoint Latin hypercube design. "Released" uses a
        Newton-type algorithm to then optimally spread the points within their assigned
        bins.

    Returns
    -------
    sample : np.ndarray
        Random Lhd.
    """
    sample = np.zeros((n_samples, dim))
    for j in range(dim):
        sample[:, j] = np.random.default_rng().choice(
            n_samples, replace=False, size=n_samples
        )

    if lhs_design == "random":
        sample += np.random.default_rng().uniform(size=sample.shape)
    elif lhs_design == "centered":
        sample += 0.5
    else:
        raise ValueError("Invalid Latin hypercube design.")

    return sample


def extend_upscaled_lhs_sample(empty_bins, target_n_points, lhs_design="random"):
    """Use previously generated points in trust region and fill out remaining bins.

    Parameters
    ----------
    emptry_bins : np.ndarray
        Array containing tuples specifying empty bins.
    target_n_points : int
        Number of points.
    lhs_design : str, optional
        One of "centered", "released". "Centered" places points in the middle of each
        bin and finds optimal midpoint Latin hypercube design. "Released" uses a
        Newton-type algorithm to then optimally spread the points within their assigned
        bins.

    Returns
    -------
    sample : np.ndarray
        Existing points extended to Lhd.
    """
    mask = empty_bins == -1
    dim = empty_bins.shape[1]
    sample = np.zeros_like(empty_bins)
    for j in range(dim):
        empty = empty_bins[:, j].copy()
        n_duplicates = mask[:, j].sum()
        empty[mask[:, j]] = np.random.default_rng().choice(
            target_n_points, replace=False, size=n_duplicates
        )
        sample[:, j] = np.random.default_rng().choice(
            empty, replace=False, size=len(empty)
        )

    if lhs_design == "random":
        sample = sample + np.random.default_rng().uniform(size=sample.shape)
    elif lhs_design == "centered":
        sample = sample + 0.5
    else:
        raise ValueError("Invalid Latin hypercube design.")

    return sample
