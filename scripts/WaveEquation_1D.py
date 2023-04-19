from PDESolver import *
from numpy import exp

def u(t, x):
    u0 = lambda _x: exp(-25 * _x**2) #1 / (200*(_x + 0.5)**2 + 1) + 1 / (200*(_x - 0.5)**2 + 2)
    return 0.5 * (u0(x - t) + u0(x + t))

# Number of iterations
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.92)
solver = Solver(WaveEquation1D(), optim, num_hidden_layers=4, num_neurons_per_layer=50)

# Train model and plot results
solver.train(iterations=N, debug_frequency=N)

error_plot_2D(solver, u, ('t', 'x'), (0, 1, -1, 1))
