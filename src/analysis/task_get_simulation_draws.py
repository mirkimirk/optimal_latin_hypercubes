"""This module contains the functions used in our solution Jupyter notebook.

neg_to_nan takes an input x of any type and converts it to a "NaN" value, if it is a
negative numerical

convert_str_to_numerical recodes strings in the survey to sensible numerical values
"""
import pickle

import numpy as np
import pytask

from src.config import BLD
from src.model_code.latin_hypercubes import optimal_latin_hypercube_sample


def draw_samples(optimality_criterion="d-optimal", lhs_design="centered", numActive=3):
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
    np.random.seed(1234)
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
    x : int or float or str
        cell value in dataframe to be converted if eligible

    Returns
    -------
    x : int or float or str
        returns NaN if x was a negative integer before, else returns the input unchanged
    """
    first_sample, second_sample = draw_samples()
    with open(produces["first"], "wb") as out_file:
        pickle.dump(first_sample, out_file)
    with open(produces["second"], "wb") as out_file:
        pickle.dump(second_sample, out_file)
