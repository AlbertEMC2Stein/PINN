from PDESolver import *


class Oscillator(BoundaryValueProblem):
    def __init__(self):
        super().__init__()
        self.initial = Cuboid([1, 0], [2, 0])
        self.initial_x = Cuboid([1, 0], [2, 0])
        self.inner = Cuboid([1, 0], [2, 6.283])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 1,
                      (self.initial, 256)),
            Condition("initial_x",
                      lambda Du: Du["u_x"],
                      (self.initial_x, 256)),
            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["t"]**2 * Du["u"],
                      (self.inner, 256))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))
            u_x = tape.gradient(u, x)
            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_x": u_x, "u_xx": u_xx}


# Number of iterations
N = 20000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(Oscillator(), num_inputs=2, num_outputs=1, num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=1000, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N)

error_plot_2D(solver, lambda t, x: np.cos(t * x), ('t', 'x'), (1, 2, 0, 6.283))
