from PDESolver import *

def u(t, x):
    u0 = lambda _x: 1 / (200*(_x + 0.5)**2 + 1) + 1 / (200*(_x - 0.5)**2 + 2)
    return 0.5 * (u0(x - t) + u0(x + t))

# Number of iterations
N = 20000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(WaveEquation1D(), num_inputs=2, num_outputs=1, num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N)

debug_plot_2D(solver, ('t', 'x'), (0, 1, -1, 1))
error_plot_2D(solver, u, ('t', 'x'), (0, 1, -1, 1))
