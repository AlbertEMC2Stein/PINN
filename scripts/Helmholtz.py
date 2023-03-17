from PDESolver import *
from numpy import pi

class HelmholtzEquation(BoundaryValueProblem):

    def __init__(self):
        super().__init__()
        self.boundary = Union(Cuboid([-1, -1], [-1, 1]), Cuboid([-1, 1], [1, 1]), Cuboid([1, 1], [1, -1]), Cuboid([1, -1], [-1, -1]))
        self.inner = Cuboid([-1, -1], [1, 1])

    def get_conditions(self):
        def q(x, y):
            return (-pi**2 - 16*pi**2 + 1) * tf.sin(pi * x) * tf.sin(4*pi * y) 

        return [
            Condition("boundary",
                      lambda Du: Du["u"],
                      (self.boundary, 128)),
            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["u_yy"] + Du["u"] - q(Du["x"], Du["y"]),
                      (self.inner, 128))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            x, y = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(x)
            tape.watch(y)

            u = model(tf.stack([x[:, 0], y[:, 0]], axis=1))
            u_x = tape.gradient(u, x)
            u_xx = tape.gradient(u_x, x)
            u_y = tape.gradient(u, y)
            u_yy = tape.gradient(u_y, y)

        del tape

        return {"x": x, "y": y, "u": u, "u_xx": u_xx, "u_yy": u_yy}


# Number of iterations
N = 40000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(HelmholtzEquation(), num_inputs=2, num_outputs=1, num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N // 2)

u = lambda t, x: tf.sin(pi * t) * tf.sin(4*pi * x)
error_plot_2D(solver, u, ('x', 'y'), (-1, 1, -1, 1))

