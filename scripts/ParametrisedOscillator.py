from PDESolver import *

class Oscillator(BoundaryValueProblem):
    def __init__(self):
        super().__init__()

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - 1,
                      (Cuboid([1, 0], [2, 0]), 128)),
            Condition("initial_x",
                      lambda Du: Du["u_x"],
                      (Cuboid([1, 0], [2, 0]), 128)),
            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["t"]**2 * Du["u"],
                      (Cuboid([1, 0], [2, 6.283]), 128))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_xx"])
                          
# Number of iterations
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.92)
solver = Solver(Oscillator(), optim, num_hidden_layers=4, num_neurons_per_layer=50)

# Train model and plot results
solver.train(iterations=N, debug_frequency=N)

error_plot_2D(solver, lambda t, x: np.cos(t * x), ('t', 'x'), (1, 2, 0, 6.283))
