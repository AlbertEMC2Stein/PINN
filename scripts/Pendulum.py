from PDESolver import *


# Number of iterations
N = 7500

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(Pendulum, num_inputs=1, num_outputs=3)
lr = tf.keras.optimizers.schedules.PolynomialDecay(0.01, N, 1e-3, 2)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N)
plot_phaseplot(solver)
