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
N = 40000

# Initialize solver
optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.92)
solver = Solver(HelmholtzEquation(), optim, num_hidden_layers=4, num_neurons_per_layer=50)

# Train model and plot results
solver.train(iterations=N, debug_frequency=N//4)

def u(t, x): return tf.sin(pi * t) * tf.sin(4*pi * x)
error_plot_2D(solver, u, ('x', 'y'), (-1, 1, -1, 1))
