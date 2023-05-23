"""This module draws two optimal Latin hypercube samples from different trust regions.

draw_samples produces two samples from different trust regions.

task_get_simulation_draws saves those samples in pickle files.
"""
import pickle

import numpy as np
import pytask

from src.config import BLD
from src.data_generator.latin_hypercubes import optimal_latin_hypercube_sample


def draw_samples(optimality_criterion="d-optimal", lhs_design="centered", numActive=5):
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
    np.random.seed(1234)
    target_n_points = 15
    first_center = np.ones(2) * 0.25
    first_radius = 0.25
    first_sample = optimal_latin_hypercube_sample(
        center=first_center,
        radius=first_radius,
        target_n_points=target_n_points,
        dim=2,
        optimality_criterion=optimality_criterion,
        lhs_design=lhs_design,
        numActive=numActive,
    )[0]

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
        dim=2,
        existing_points=existing,
        optimality_criterion=optimality_criterion,
        lhs_design=lhs_design,
        numActive=numActive,
    )[0]

    full_region, F_crit, crit_val_list = optimal_latin_hypercube_sample(15, 2)
    full_region2, F_crit2, crit_val_list2 = optimal_latin_hypercube_sample(15, 2)

    return (
        first_sample,
        second_sample,
        full_region,
        full_region2,
        crit_val_list,
        crit_val_list2,
    )


@pytask.mark.produces(
    {
        "first": BLD / "data" / "first.pickle",
        "second": BLD / "data" / "second.pickle",
        "full": BLD / "data" / "full.pickle",
        "full2": BLD / "data" / "full2.pickle",
        "crit_val_list": BLD / "data" / "crit_val_list.pickle",
        "crit_val_list2": BLD / "data" / "crit_val_list2.pickle",
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
    sample_names = list(produces.keys())
    samples = draw_samples()
    for x, y in zip(sample_names, samples):
        globals()[x] = y
    for i in sample_names:
        with open(produces[i], "wb") as out_file:
            pickle.dump(eval(i), out_file)
