from PDESolver import *


# Number of iterations
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(Pendulum, num_inputs=1, num_outputs=3, num_hidden_layers=5, num_neurons_per_layer=16)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, N, 1e-4)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N)

plot_phaseplot(solver)
