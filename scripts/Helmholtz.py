from PDESolver import *
from numpy import pi

class HelmholtzEquation(BoundaryValueProblem):
    def __init__(self):
        super().__init__()

        def q(x, y): return (-pi**2 - 16*pi**2 + 1) * tf.sin(pi * x) * tf.sin(4*pi * y)

        self.conditions = [
            Condition("boundary",
                      lambda Du: Du["u"],
                      (Cuboid([-1, -1], [-1, 1]) & Cuboid([-1, 1], [1, 1]) & Cuboid([1, 1], [1, -1]) & Cuboid([1, -1], [-1, -1]), 128)),

            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["u_yy"] + Du["u"] - q(Du["x"], Du["y"]),
                      (Cuboid([-1, -1], [1, 1]), 128))
        ]

        self.specification = Specification(["u"], ["x", "y"], ["u_xx", "u_yy"])

# Number of iterations
N = 1000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(HelmholtzEquation(), num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, N, N)

def u(t, x): return tf.sin(pi * t) * tf.sin(4*pi * x)
error_plot_2D(solver, u, ('x', 'y'), (-1, 1, -1, 1))
