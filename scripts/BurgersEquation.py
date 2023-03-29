from PDESolver import *

# Number of iterations
N = 20000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(BurgersEquation(viscosity=0.0), num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=250, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N)

debug_plot_2D(solver, ('t', 'x'), (0, 1, -1, 1))
