from PDESolver import *
from scipy.integrate import quad

# Number of iterations
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
optim = Optimizer(initial_learning_rate=0.01, annealing_factor=0.9)
solver = Solver(Laplace(), optim, num_hidden_layers=4, num_neurons_per_layer=50)

# Train model and plot results
solver.train(iterations=N, debug_frequency=N)

# Math: G_n = \frac{2}{\sinh\left(N\pi\right)}\int_{0}^{1}2t\left(1-t\right)\sin\left(N\pi t\right)dt
# Math: u(x, y) = \sum_{n=1}^{k}G\left[n\right]\sinh\left(n\pi y\right)\sin\left(n\pi x\right)
u1 = lambda t: 2 * t * (1 - t)
coeffs = [2 / np.sinh(np.pi * n) * quad(lambda t: u1(t) * np.sin(n * np.pi * t), 0, 1)[0] for n in range(1, 52)]
u = lambda x, y: sum([coeffs[n - 1] * np.sinh(n * np.pi * y) * np.sin(n * np.pi * x) for n in range(1, 52)])

error_plot_2D(solver, u, ('x', 'y'), (0, 1, 0, 1))
