from src.model_code.latin_hypercubes import optimal_latin_hypercube_sample


def test_main_function():
    optimal_latin_hypercube_sample(
        5,
        S=None,
        center=None,
        radius=0.5,
        existing_points=None,
        optimality_criterion="d-optimal",
        lhs_design="centered",
        numActive=3,
    )
