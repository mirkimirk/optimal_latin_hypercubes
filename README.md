# Optimal Latin Hypercubes
This project is about implementing Park's (1994) algorithm for Optimal Latin Hypercube
sampling in a python function.

The documentation of the code is in src/documentation. To run the code, use pytask
in the shell – the results will be in the bld folder.

The heart of the project is latin_hypercubes.py, situated in src/model_code. It contains
the function to produce an optimal Latin hypercube, following Park's (1994) algorithm.
In a first stage, the algorithm finds an optimal midpoint Latin hypercube design (OMLhd).
In the second stage, it releases the points within each cell optimally to produce the
optimal Latin hypercube design (OLhd). (Second stage still needs to be implemented.)