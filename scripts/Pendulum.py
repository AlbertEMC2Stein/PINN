from PDESolver import *

# Number of iterations
N = 40000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(Pendulum(), num_inputs=1, num_outputs=3, num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N)

plot_phaseplot(solver)
