# thomson-problem-de

Differential Evolution optimization used to find candidate solutions for the Thomson problem.

Thomson problem aims to find configuration for N charges confined to the surface of a sphere where the
potential is at a minimum. Potential in this case is calculated use columb's potential.

## Detail on algorithm

The algorithm uses only selection and mutation. It switches between a standard mutation strategy and a strategy with jitter depending on whether the standard deviation of the previous 10 generations have changed or not.

Output plot shows the evaluated potential for the best and worst sphere in each generation and upon completion of the program, the best overall sphere is plotted with its corresponding configuration.
