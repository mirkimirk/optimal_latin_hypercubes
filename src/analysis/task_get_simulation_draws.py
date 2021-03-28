"""This module draws two optimal Latin hypercube samples from different trust regions.

draw_samples produces two samples from different trust regions.

task_get_simulation_draws saves those samples in pickle files.
"""
import pickle

import numpy as np
import pytask

from src.config import BLD
from src.model_code.latin_hypercubes import optimal_latin_hypercube_sample


def draw_samples(optimality_criterion="d-optimal", lhs_design="centered", numActive=3):
    """Draw two specified samples from different trust regions for later plotting.

    Parameters
    ----------
    optimality_criterion : str
        One of "a-optimal", "d-optimal", "e-optimal",
        "t-optimal", "g-optimal".
    lhs_design : str
        One of "centered", "released". "Centered" places points in the middle of each
        bin and finds optimal midpoint Latin hypercube design. "Released" uses a
        Newton-type algorithm to then optimally spread the points within their assigned
        bins.
    numActive : int
        Number of row pairs of S to build and use for exchanging values in their
        columns.

    Returns
    -------
    first_sample : np.ndarray
        returns NaN if x was a negative integer before, else returns the input unchanged
    second_sample : np.ndarray
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    np.random.seed(12345)
    target_n_points = 10
    first_center = np.ones(2) * 0.25
    first_radius = 0.25
    first_sample = optimal_latin_hypercube_sample(
        center=first_center,
        radius=first_radius,
        target_n_points=target_n_points,
        optimality_criterion=optimality_criterion,
        lhs_design=lhs_design,
        numActive=numActive,
    )[1]

    second_center = np.ones(2) * 0.4
    second_radius = 0.25
    lower = second_center - second_radius
    upper = second_center + second_radius
    existing = []
    for row in first_sample:
        if (lower <= row).all() and (upper >= row).all():
            existing.append(row)

    existing = np.array(existing)

    second_sample = optimal_latin_hypercube_sample(
        center=second_center,
        radius=second_radius,
        target_n_points=target_n_points,
        existing_points=existing,
        optimality_criterion=optimality_criterion,
        lhs_design=lhs_design,
        numActive=numActive,
    )[1]

    return first_sample, second_sample


@pytask.mark.produces(
    {
        "first": BLD / "data" / "first_sample.pickle",
        "second": BLD / "data" / "second_sample.pickle",
    }
)
def task_get_simulation_draws(produces):
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    produces : path
        Variable for pytask. Specifies locations to output the samples to.

    Returns
    -------
    first_sample : np.ndarray
        returns NaN if x was a negative integer before, else returns the input unchanged
    second_sample : np.ndarray
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    first_sample, second_sample = draw_samples()
    with open(produces["first"], "wb") as out_file:
        pickle.dump(first_sample, out_file)
    with open(produces["second"], "wb") as out_file:
        pickle.dump(second_sample, out_file)
