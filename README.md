# Optimal Latin Hypercubes
This project is about implementing Park's (1994) algorithm for Optimal Latin Hypercube
sampling in a python function.

## NOTE:
This project is not actively worked on right now (May 2023). Until now only the
two steps of the first stage of Park's algorithm are implemented. To Dos:
- Cleaning up the code, modularization
- More presentation of sample generation and convergence rates of the algorithm
- Tests


The documentation of the code is in src/documentation. To run the code, use pytask
in the shell – the results will be in the bld folder.

The heart of the project is latin_hypercubes.py, situated in src/model_code. It contains
the function to produce an optimal Latin hypercube, following Park's (1994) algorithm.
In a first stage, the algorithm finds an optimal midpoint Latin hypercube design (OMLhd).
In the second stage, it releases the points within each cell optimally to produce the
optimal Latin hypercube design (OLhd). (Second stage still needs to be implemented.)

(Big parts of the code, especially in "latin_hypercubes_aux.py", were "inspired" by an
implementation of Lhd sampling from another Python package, but I don't remember which
one. I think SciPy.)

## References
Park, J.-S. (1994). Optimal Latin-hypercube designs for computer experiments. _Journal of Statistical Planning and Inference_, _39_(1), 95–111. [https://doi.org/10.1016/0378-3758(94)90115-5](https://doi.org/10.1016/0378-3758\(94\)90115-5)
