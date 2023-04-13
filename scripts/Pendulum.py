from PDESolver import *

# Number of iterations
N = 1000

# Initialize solver, learning rate scheduler and choose optimizer
t_start = 0.
t_mid = 5.
t_end = 10.

solver01 = Solver(Pendulum(t_start=t_start, t_end=t_mid), num_hidden_layers=6, num_neurons_per_layer=16)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train first section model
solver01.train(optim, N, -1)

# Determine initial conditions for next section
data = solver01.compute_differentials(tf.constant([[t_mid]]))
init_pos = data['u'].numpy()[0]
init_vel = data['u_t'].numpy()[0]
print("Initial position: ", init_pos, "Initial velocity: ", init_vel)

# Train second section model
solver12 = Solver(Pendulum(init_pos, init_vel, t_start=t_mid, t_end=t_end), num_hidden_layers=6, num_neurons_per_layer=16)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver12.train(optim, N, -1)

# Combine models
class Connect:
    def __init__(self, solver1, solver2, t_intermediate):
        self.solver1 = solver1
        self.solver2 = solver2
        self.bvp = solver1.bvp

        self.model = lambda x: tf.where(x < t_intermediate, solver1.model(x), solver2.model(x))
        self.loss_history = np.mean(np.vstack([solver1.loss_history, solver2.loss_history]), axis=0) 

# Plot results
plot_phaseplot(Connect(solver01, solver12, t_mid), t_start, t_end)
