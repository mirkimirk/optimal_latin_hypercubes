"""This module contains functions to search an OMLhd sample with Park's (1994) algorithm.

The function optimal_latin_hypercube_sample uses Park's (1994) algorithm to find an
optimal (midpoint) Latin hypercube design. At this point, only the first stage of the
algorithm – finding an optimal midpoint Latin hypercube – is implemented. The second
stage of the algorithm optimally releases each point within its assigned bin. (The
second stage is not yet implemented.)
"""
from itertools import combinations

import numpy as np

# Potential objective functions to serve as criterion of quality of Lhd sample. a-, d-,
# e- and t-optimality functions emphasize in-sample fitting usefulness, while
# g-optimality emphasizes usefulness for prediction.
OPTIMALITY_CRITERIA = {
    "a-optimal": lambda x: np.linalg.inv(x.T @ x).trace(),
    "d-optimal": lambda x: np.linalg.det(np.linalg.inv(x.T @ x)),
    "e-optimal": lambda x: np.amin(np.linalg.eig(x.T @ x)[0])
    * (-1),  # argmax of minimal eigenvalue = argmin of negative minimal eigenvalue
    "t-optimal": lambda x: np.trace(x.T @ x) * (-1),
    "g-optimal": lambda x: np.amax(np.diagonal(x @ np.linalg.inv(x.T @ x) @ x.T)),
}


def optimal_latin_hypercube_sample(
    target_n_points,
    S_init=None,
    center=np.ones(2) * 0.5,
    radius=0.5,
    existing_points=None,
    optimality_criterion="d-optimal",
    lhs_design="centered",
    numActive=3,
    numRand=None,
    n_iter=100,
):
    """Generate the optimal (midpoint) Latin hypercube sample (O(M)Lhd).

    This function implements Park's (1994) algorithm for finding an O(M)Lhd either from
    a supplied MLhd, or a randomly generated one. The region of interest is the unit
    cube per default. The algorithm consists of two stages: finding an OMLhd, and in the
    second stage to find an optimal Latin hypercube by releasing the points within their
    optimal bins. (The second stage is not yet implemented).

    In the first stage, two steps are performed: first, the impact of a given number of
    row pairs (rows of the sample S) on the objective function is evaluated, and the
    pairs with the most optimizing impact are selected. In the second step, all
    potential (even-numbered) combinations of columns are evaluated by exchanging their
    values in the given two rows. This is done for all active pairs of rows and repeated
    until the objective function does not change anymore.

    Parameters
    ----------
    target_n_points : int
        Target number of points in the trust region at which criterion
        values are known. The actual number can be larger than this
        if the existing points are badly spaced.
    S_init : np.ndarray, optional
        An existing midpoint Latin hypercube sample in the region of interested that can
        be used as a starting point for the algorithm. If not specified, a random Lhd
        is generated instead as a starting point.
    center : one-dimensional np.ndarray, optional
        Center of the current trust region. Default is a vector of length 2 with both
        values set to 0.5.
    radius : float, optional
        Radius of the current trust region.
    existing_points : np.ndarray, optional
        2d Array where each row is a parameter vector at which
        the criterion has already been evaluated. This is ignored, if S_0 is provided.
    optimality_criterion : str, optional
        One of "a-optimal", "d-optimal", "e-optimal",
        "t-optimal", "g-optimal".
    lhs_design : str, optional
        One of "centered", "released". "Centered" places points in the middle of each
        bin and finds optimal midpoint Latin hypercube design. "Released" uses a
        Newton-type algorithm to then optimally spread the points within their assigned
        bins.
    numActive : int, optional
        Number of row pairs of S to build and use for exchanging values in their
        columns.
    numRand : int, optional
        Number of random row pairs to try after successful finish of first stage to
        reduce risk of finding a non-optimal OMLhd. Default is 3 * target_n_points // 5.
    n_iter : int, optional
        Number of random MLhds to try out as a starting point of the algorithm.

    Returns
    -------
    f : float
        Value of objective function if optimal sample is evaluated.
    S : np.ndarray
        Optimal (midpoint) Latin hypercube sample.
    """
    criterion_func = OPTIMALITY_CRITERIA[optimality_criterion]
    dim = len(center)
    if numRand is None:
        numRand = 3 * target_n_points // 5

    f_candidates = []
    S_candidates = []
    for _ in range(n_iter):
        S_0 = S_init
        if existing_points is None:
            if S_0 is None:
                upscaled_points = _create_upscaled_lhs_sample(
                    dim=dim,
                    n_samples=target_n_points,
                    lhs_design="centered",
                )
                S_0 = _scale_down_points(
                    upscaled_points, center, radius, target_n_points
                )

        else:
            if S_0 is None:
                existing_upscaled = _scale_up_points(
                    existing_points,
                    center,
                    radius,
                    target_n_points,
                )
                empty_bins = _get_empty_bin_info(existing_upscaled, target_n_points)
                new = _extend_upscaled_lhs_sample(
                    empty_bins=empty_bins,
                    target_n_points=target_n_points,
                    lhs_design="centered",
                )
                upscaled_points = np.row_stack([existing_upscaled, new])
                S_0 = _scale_down_points(
                    upscaled_points, center, radius, target_n_points
                )

        # Step 1
        f_0, active_pairs = _step_1(
            criterion_func=criterion_func, S=S_0, numActive=numActive
        )

        # Step 2
        f_1, S_1 = _step_2(
            criterion_func=criterion_func,
            dim=dim,
            S=S_0,
            crit_val=f_0,
            n_pairs=numActive,
            active_pairs=active_pairs,
        )

        # Double check with random pairs that are not the same as the active pairs
        potential_random_pairs = np.array(
            list(set(combinations(range(target_n_points), 2)) - set(active_pairs))
        )
        random_pairs = np.random.default_rng().choice(
            a=potential_random_pairs,
            size=numRand,
            replace=False,
            axis=0,
            shuffle=False,  # saves some time
        )
        f_2, S_2 = _step_2(
            criterion_func=criterion_func,
            dim=dim,
            S=S_1,
            crit_val=f_1,
            n_pairs=numRand,
            active_pairs=random_pairs,
        )

        f_candidates.append(f_2)
        S_candidates.append(S_2)

    best = np.argmin(f_candidates)

    f = f_candidates[best]
    S = S_candidates[best]

    # Second stage of algorithm - Newton-type optimization
    if lhs_design == "released":
        pass

    return f, S


def _step_1(criterion_func, S, numActive):
    """
    Implement the first step of the first stage of Park's (1994) algorithm for OMLHD.

    Parameters
    ----------
    criterion_func : function
        Objective function used to evaluate the quality of the Lhd samples.
    S : np.ndarray
        Initial midpoint Latin hypercube sample to modify by the algorithm.
    numActive : int
        Number of row pairs of S to build and use for exchanging values in their
        columns.

    Returns
    -------
    crit_val : float
        Value of objective function if initial sample is evaluated.
    active_pairs : list of tuples
        List of active row pairs to evaluate in second step.
    """
    crit_val = criterion_func(S)
    n = len(S)
    function_values = np.empty(n)
    for i in range(n):
        function_values[i] = criterion_func(np.delete(arr=S, obj=i, axis=0))
    pairing_candidates = np.argsort(function_values)
    # take the first numActive combinations of the first numActive pairing_candidates
    active_pairs = list(combinations(pairing_candidates[:numActive], 2))[:numActive]

    return crit_val, active_pairs


def _step_2(criterion_func, dim, S, crit_val, n_pairs, active_pairs):
    """
    Implement the second step of the first stage of Park's (1994) algorithm for OMLHD.

    Parameters
    ----------
    criterion_func : function
        Objective function used to evaluate the quality of the Lhd samples.
    dim : int
        Number of variables in the design space.
    S : np.ndarray
        Initial midpoint Latin hypercube sample to modify by the algorithm.
    crit_val : float
        Value of objective function if initial sample is evaluated.
    n_pairs : int
        Number of row pairs of S to build and use for exchanging values in their
        columns.
    active_pairs : list of tuples
        List of row pairs to evaluate in second step.

    Returns
    -------
    f_1 : float
        Value of objective function if optimal sample is evaluated.
    S : np.ndarray
        Optimal (midpoint) Latin hypercube sample.
    """
    it = (2 ** (dim - 1)) - 1
    n_components = np.where(np.arange(dim + 1) % 2 == 0)[0][1:]
    # loop over even-numbered combinations
    switching_components_list = [
        list(combinations(range(dim), i)) for i in n_components
    ]
    # for i in n_components:
    #     switching_components_list.append(list(combinations(range(dim), i)))
    # flatten list of lists
    switching_components = [
        item for sublist in switching_components_list for item in sublist
    ]
    i = 0
    while i < n_pairs:
        active_pair = active_pairs[i]
        first_row = active_pair[0]
        second_row = active_pair[1]
        function_values_step2 = np.empty(it)
        for j in range(it):
            switching_component = switching_components[j]
            S_temp = S.copy()
            S_temp[([first_row], [second_row]), switching_component] = S[
                ([second_row], [first_row]), switching_component
            ]
            function_values_step2[j] = criterion_func(S_temp)
        winning_switch = switching_components[np.argmin(function_values_step2)]
        S_temp = S.copy()
        S_temp[([first_row], [second_row]), winning_switch] = S[
            ([second_row], [first_row]), winning_switch
        ]
        if criterion_func(S_temp) < crit_val:
            S = S_temp
            crit_val, active_pairs = _step_1(
                criterion_func=criterion_func, S=S, numActive=n_pairs
            )
            i = 0
            continue
        else:
            i += 1

    return crit_val, S


def _scale_up_points(points, center, radius, target_n_points):
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


def _scale_down_points(points, center, radius, target_n_points):
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


def _get_empty_bin_info(existing_upscaled, target_n_points):
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


def _create_upscaled_lhs_sample(dim, n_samples, lhs_design="centered"):
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
        sample[:, j] = np.random.choice(n_samples, replace=False, size=n_samples)

    if lhs_design == "random":
        sample += np.random.uniform(size=sample.shape)
    elif lhs_design == "centered":
        sample += 0.5
    else:
        raise ValueError("Invalid latin hypercube design.")

    return sample


def _extend_upscaled_lhs_sample(empty_bins, target_n_points, lhs_design="random"):
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
