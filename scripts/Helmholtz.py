from PDESolver import *
from numpy import pi

class HelmholtzEquation(BoundaryValueProblem):
    def __init__(self):
        super().__init__()
        self.boundary = Union(Cuboid([-1, -1], [-1, 1]), Cuboid([-1, 1], [1, 1]), Cuboid([1, 1], [1, -1]), Cuboid([1, -1], [-1, -1]))
        self.inner = Cuboid([-1, -1], [1, 1])

        q = lambda x, y: (-pi**2 - 16*pi**2 + 1) * tf.sin(pi * x) * tf.sin(4*pi * y)

        self.c1 = Condition("boundary",
                      lambda Du: Du["u"],
                      (self.boundary, 128))
        
        self. c2= Condition("inner",
                      lambda Du: Du["u_xx"] + Du["u_yy"] + Du["u"] - q(Du["x"], Du["y"]),
                      (self.inner, 128))

    def get_conditions(self):
        return [self.c1, self.c2]

    def get_specification(self):
        return {
            "components": ["u"],
            "variables": ["x", "y"],
            "differentials": ["u_xx", "u_yy"],
        }


# Number of iterations
N = 5000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(HelmholtzEquation(), num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N)

u = lambda t, x: tf.sin(pi * t) * tf.sin(4*pi * x)
error_plot_2D(solver, u, ('x', 'y'), (-1, 1, -1, 1))

