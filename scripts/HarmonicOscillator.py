from PDESolver import *

class HarmonicOscillator(BoundaryValueProblem):
    def __init__(self):
        super().__init__()

        self.conditions = [
            Condition(
                "initial",
                lambda Du: Du["u"],
                (Cuboid([0], [0]), 1),
            ),
            Condition(
                "initial_x",
                lambda Du: Du["u_x"] - 1,
                (Cuboid([0], [0]), 1),
            ),
            Condition(
                "inner",
                lambda Du: Du["u_xx"] + Du["u"],
                (Cuboid([0], [6.283]), 50),
            ),
        ]

        self.specification = Specification(["u"], ["x"], ["u_xx", "u_x"])

# Number of iterations
N = 5000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(HarmonicOscillator(), num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, N, 0.01)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N, N)

x = tf.linspace(0.0, 6.283, 600, axis=0)
y = solver.model(x)

plt.plot(x, np.sin(x), "b")
plt.plot(x, y, "r--")
plt.grid()
plt.xlim(0, 6.283)
plt.plot([0, 7], [0, 0], lw=1, c="k")
plt.show()
