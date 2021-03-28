"""This module contains the functions used in our solution Jupyter notebook.

neg_to_nan takes an input x of any type and converts it to a "NaN" value, if it is a
negative numerical

convert_str_to_numerical recodes strings in the survey to sensible numerical values
"""
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import pytask
import pickle

from src.config import BLD


@pytask.mark.depends_on(
    {
        "first": BLD / "data" / "first_sample.pickle",
        "second": BLD / "data" / "second_sample.pickle",
    }
)
@pytask.mark.produces(BLD / "figures" / "plot.pdf")
def task_plot(
    depends_on, produces, optimality_criterion="a-optimal", lhs_design="centered"
):
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
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    first_sample = pickle.load(open(depends_on["first"], "rb"))
    second_sample = pickle.load(open(depends_on["second"], "rb"))
    sns.regplot(
        x=first_sample[:, 0],
        y=first_sample[:, 1],
        ax=ax,
        fit_reg=False,
        color="darkblue",
        scatter_kws={"alpha": 0.4},
    )
    sns.regplot(
        x=second_sample[:, 0],
        y=second_sample[:, 1],
        ax=ax,
        fit_reg=False,
        color="firebrick",
        scatter_kws={"alpha": 0.4},
    )

    for i in np.arange(0, 1, 1 / 20):
        plt.axhline(i)
        plt.axvline(i)

    plt.savefig(produces)
