from PDESolver import *

# Number of iterations
N = 20000

# Initialize solver
optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.92)
solver = Solver(BurgersEquation(viscosity=0.01), optim, num_hidden_layers=4, num_neurons_per_layer=50)

# Train model and plot results
solver.train(iterations=N, debug_frequency=N)

debug_plot_2D(solver, ('t', 'x'), (0, 1, -1, 1))
