"""This module contains functions to search an OMLhd sample with Park's (1994) algorithm.

The function optimal_latin_hypercube_sample uses Park's (1994) algorithm to find an
optimal (midpoint) Latin hypercube design. At this point, only the first stage of the
algorithm – finding an optimal midpoint Latin hypercube – is implemented. The second
stage of the algorithm, i.e., optimally releasing each point within its assigned bin,
will be implemented later.

neg_to_nan takes an input x of any type and converts it to a "NaN" value, if it is a
negative numerical

convert_str_to_numerical recodes strings in the survey to sensible numerical values
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
    center=None,
    radius=0.5,
    existing_points=None,
    optimality_criterion="d-optimal",
    lhs_design="centered",
    numActive=3,
):
    """Generate new points at which the criterion should be evaluated.

    This function implements

    Parameters
    ----------
    target_n_points : int
        Target number of points in the trust region at which criterion
        values are known. The actual number can be larger than this
        if the existing points are badly spaced.
    center : np.ndarray
        Center of the current trust region.
    radius : float
        Radius of the current trust region.
    existing_points : np.ndarray
        2d Array where each row is a parameter vector at which
        the criterion has already been evaluated.
    optimality_criterion : str
        One of "a-optimal", "d-optimal", "e-optimal",
        "t-optimal", "g-optimal".
    lhs_design : str
        One of "centered", "released". "Centered" places points in the middle of each
        bin and finds optimal midpoint Latin hypercube design. "Released" uses a
        Newton-type algorithm to then optimally spread the points within their assigned
        bins.
    Returns
    -------
    f_0 : float
        Value of objective function
    """
    if center is None:
        center = np.ones(target_n_points) * 0.5
    criterion_func = OPTIMALITY_CRITERIA[optimality_criterion]
    dim = len(center)

    if existing_points is None:
        upscaled_points = _create_upscaled_lhs_sample(
            dim=dim,
            n_samples=target_n_points,
            lhs_design="centered",
        )
        S = _scale_down_points(upscaled_points, center, radius, target_n_points)
        # Step 1
        f_0, active_pairs = _step_1(
            criterion_func=criterion_func, S=S, numActive=numActive
        )
        # Step 2
        f_0, S = _step_2(
            criterion_func=criterion_func,
            dim=dim,
            S=S,
            f_0=f_0,
            numActive=numActive,
            active_pairs=active_pairs,
        )

    else:
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
        S = _scale_down_points(upscaled_points, center, radius, target_n_points)
        # Step 1
        f_0, active_pairs = _step_1(
            criterion_func=criterion_func, S=S, numActive=numActive
        )
        # Step 2
        f_0, S = _step_2(
            criterion_func=criterion_func,
            dim=dim,
            S=S,
            f_0=f_0,
            numActive=numActive,
            active_pairs=active_pairs,
        )
    if lhs_design == "released":
        pass

    return f_0, S


def _step_1(criterion_func, S, numActive):
    """
    Implement the first step of the first stage of Park's (1994) algorithm for OMLHD.

    Parameters
    ----------
    criterion_func : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    f_0 = criterion_func(S)
    n = len(S)
    function_values = np.empty(n)
    for i in range(n):
        function_values[i] = criterion_func(np.delete(arr=S, obj=i, axis=0))
    pairing_candidates = np.argsort(function_values)[:numActive]
    # take the first numActive combinations of the pairing_candidates
    active_pairs = list(combinations(pairing_candidates, 2))[:numActive]

    return f_0, active_pairs


def _step_2(criterion_func, dim, S, f_0, numActive, active_pairs):
    """
    Implement the second step of the first stage of Park's (1994) algorithm for OMLHD.

    Parameters
    ----------
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    it = (2 ** (dim - 1)) - 1
    n_components = np.where(np.arange(dim + 1) % 2 == 0)[0][1:]
    switching_components_list = []
    # loop over even-numbered combinations
    for i in n_components:
        switching_components_list.append(list(combinations(range(dim), i)))
    # flatten list of lists
    switching_components = [
        item for sublist in switching_components_list for item in sublist
    ]
    i = 0
    while i < numActive:
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
        if criterion_func(S_temp) < f_0:
            S = S_temp
            f_0, active_pairs = _step_1(
                criterion_func=criterion_func, S=S, numActive=numActive
            )
            continue
        else:
            i += 1

    return f_0, S


def _scale_up_points(points, center, radius, target_n_points):
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    lower = center - radius
    scaled = (points - lower) * target_n_points / (2 * radius)
    return scaled


def _scale_down_points(points, center, radius, target_n_points):
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    lower = center - radius
    scaled = points / target_n_points * (2 * radius) + lower
    return scaled


def _get_empty_bin_info(existing_upscaled, target_n_points):
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
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


def _create_upscaled_lhs_sample(dim, n_samples, lhs_design="random"):
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
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
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
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


target_n_points = 10
first_center = np.ones(2) * 0.25
first_radius = 0.25
optimality_criterion = "d-optimal"
lhs_design = "centered"
numActive = 3
first_sample = optimal_latin_hypercube_sample(
    center=first_center,
    radius=first_radius,
    target_n_points=target_n_points,
    optimality_criterion=optimality_criterion,
    lhs_design=lhs_design,
    numActive=numActive,
)[1]
