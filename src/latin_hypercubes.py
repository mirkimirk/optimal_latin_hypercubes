from itertools import combinations

import numpy as np


OPTIMALITY_CRITERIA = {
    "a-optimal": lambda x: np.linalg.inv(x.T @ x).trace(),
    "d-optimal": lambda x: np.linalg.det(np.linalg.inv(x.T @ x)),
    "e-optimal": lambda x: np.amin(np.linalg.eig(x.T @ x)[0])
    * (-1),  # argmax of minimal eigenvalue = argmin of negative minimal eigenvalue
    "t-optimal": lambda x: np.trace(x.T @ x) * (-1),
    "g-optimal": lambda x: np.amax(np.diagonal(x @ np.linalg.inv(x.T @ x) @ x.T)),
    "stupid": lambda x: x[0, 0],
}


def get_next_trust_region_points_latin_hypercube(
    target_n_points,
    center=0.5,
    radius=0.5,
    existing_points=None,
    optimality_criterion="a-optimal",
    lhs_design="centered",
    n_iter=100,
    numActive=3,
):
    """Generate new points at which the criterion should be evaluated.

    Args:
        center (np.ndarray): Center of the current trust region.
        radius (float): Radius of the current trust region.
        target_n_points (int): Target number of points in the trust
            region at which criterion values are known. The actual
            number can be larger than this if the existing points
            are badly spaced.
        existing_points (np.ndarray): 2d Array where each row is a
            parameter vector at which the criterion has already been
            evaluated.
        optimality_criterion (str): One of "a-optimal", "d-optimal",
            "e-optimal", "t-optimal", "g-optimal".
        lhs_design (str): One of "random", "centered".
            determines how sample points are spaced inside bins.

    """
    criterion_func = OPTIMALITY_CRITERIA[optimality_criterion]
    dim = len(center)

    if existing_points is None:
        current_best = np.inf
        S = _create_upscaled_lhs_sample(
            dim=dim,
            n_samples=target_n_points,
            lhs_design=lhs_design,
        )
        f_0 = criterion_func(S)
        function_values = np.empty(target_n_points)
        for i in range(target_n_points):
            function_values[i] = criterion_func(np.delete(S, 0, i))
        pairing_candidates = np.argsort(function_values)[:numActive]
        active_pairs = list(combinations(pairing_candidates), 2)
        f_0
        active_pairs
        for _ in range(n_iter):
            candidate = _create_upscaled_lhs_sample(
                dim=dim,
                n_samples=target_n_points,
                lhs_design=lhs_design,
            )
            crit_val = criterion_func(candidate)
            if crit_val < current_best:
                current_best = crit_val
                upscaled_points = candidate

    else:
        existing_upscaled = _scale_up_points(
            existing_points,
            center,
            radius,
            target_n_points,
        )
        empty_bins = _get_empty_bin_info(existing_upscaled, target_n_points)

        current_best = np.inf
        for _ in range(n_iter):
            new = _extend_upscaled_lhs_sample(
                empty_bins=empty_bins,
                target_n_points=target_n_points,
                lhs_design=lhs_design,
            )
            candidate = np.row_stack([existing_upscaled, new])
            crit_val = criterion_func(candidate)
            if crit_val < current_best:
                current_best = crit_val
                upscaled_points = candidate

    points = _scale_down_points(upscaled_points, center, radius, target_n_points)

    return points


def _scale_up_points(points, center, radius, target_n_points):
    lower = center - radius
    scaled = (points - lower) * target_n_points / (2 * radius)
    return scaled


def _scale_down_points(points, center, radius, target_n_points):
    lower = center - radius
    scaled = points / target_n_points * (2 * radius) + lower
    return scaled


def _get_empty_bin_info(existing_upscaled, target_n_points):
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


def _create_upscaled_lhs_sample(dim, n_samples, lhs_design="random"):
    sample = np.zeros((n_samples, dim))
    for j in range(dim):
        sample[:, j] = np.random.choice(n_samples, replace=False, size=n_samples)

    if lhs_design == "random":
        sample += np.random.uniform(size=sample.shape)
    elif lhs_design == "centered":
        sample += 0.5
    else:
        raise ValueError("Invalid latin hypercube design.")

    return sample


def _extend_upscaled_lhs_sample(empty_bins, target_n_points, lhs_design="random"):
    mask = empty_bins == -1  # whats this for? "-1"?
    dim = empty_bins.shape[1]
    sample = np.zeros_like(empty_bins)
    for j in range(dim):
        empty = empty_bins[:, j].copy()
        n_duplicates = mask[:, j].sum()  # whats this for?
        empty[mask[:, j]] = np.random.choice(
            target_n_points, replace=False, size=n_duplicates
        )
        sample[:, j] = np.random.choice(empty, replace=False, size=len(empty))

    if lhs_design == "random":
        sample = sample + np.random.uniform(size=sample.shape)
    elif lhs_design == "centered":
        sample = sample + 0.5
    else:
        raise ValueError("Invalid latin hypercube design.")

    return sample
