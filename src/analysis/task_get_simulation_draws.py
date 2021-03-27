"""

"""
import pickle

import numpy as np
import pytask

from src.config import BLD
from src.latin_hypercubes import get_next_trust_region_points_latin_hypercube


def draw_samples(optimality_criterion="a-optimal", lhs_design="centered"):
    np.random.seed(1234)
    target_n_points = 10
    first_center = np.ones(2) * 0.25
    first_radius = 0.25
    first_sample = get_next_trust_region_points_latin_hypercube(
        center=first_center,
        radius=first_radius,
        target_n_points=target_n_points,
        n_iter=1000,
        optimality_criterion=optimality_criterion,
        lhs_design=lhs_design,
    )

    second_center = np.ones(2) * 0.4
    second_radius = 0.25
    lower = second_center - second_radius
    upper = second_center + second_radius
    existing = []
    for row in first_sample:
        if (lower <= row).all() and (upper >= row).all():
            existing.append(row)

    existing = np.array(existing)

    second_sample = get_next_trust_region_points_latin_hypercube(
        center=second_center,
        radius=second_radius,
        target_n_points=target_n_points,
        n_iter=1000,
        existing_points=existing,
        optimality_criterion=optimality_criterion,
        lhs_design=lhs_design,
    )

    return first_sample, second_sample


@pytask.mark.produces(
    {
        "first": BLD / "data" / "first_sample.pickle",
        "second": BLD / "data" / "second_sample.pickle",
    }
)
def task_get_simulation_draws(produces):
    first_sample, second_sample = draw_samples()
    with open(produces["first"], "wb") as out_file:
        pickle.dump(first_sample, out_file)
    with open(produces["second"], "wb") as out_file:
        pickle.dump(second_sample, out_file)
