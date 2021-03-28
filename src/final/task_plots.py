"""This module produces a plot illustrating a possible trust region application.

task_plot is a function that takes as inputs the location of sample data and where to
output, and yields a plot.
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
def task_plot(depends_on, produces):
    """Convert negative numbers from string to np.nan.

    Parameters
    ----------
    depends_on : path
        Variable for pytask. Specifies dependency, i.e., the data to plot.
    produces : path
        Variable for pytask. Specifies location to output the plot to.

    Returns
    -------
    fig : figure
        A plot showing a possible trust region application.
    """
    fig, ax = plt.subplots()
    fig.suptitle("Illustration of trust region application with reused sample points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("$F(x_2)$")
    ax.set_xlabel("$F(x_1)$")

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
