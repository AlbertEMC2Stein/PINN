from PDESolver import *


class HarmonicOscillator(BoundaryValueProblem):
    @classproperty
    def conditions(cls):
        return [
            Condition(
                "initial",
                lambda Du: Du["u"],
                (Cuboid([0], [0]), 1),
                Equidistant()
            ),
            Condition(
                "initial_x",
                lambda Du: Du["u_x"] - 1,
                (Cuboid([0], [0]), 1),
                Equidistant()
            ),
            Condition(
                "inner",
                lambda Du: Du["u_xx"] + Du["u"],
                (Cuboid([0], [6.283]), 20),
                Equidistant()
            ),
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            x = freeVariables[:, 0:1]

            tape.watch(x)

            u = model(x)
            u_x = tape.gradient(u, x)
            u_xx = tape.gradient(u_x, x)
        del tape

        return {"x": x, "u": u, "u_x": u_x, "u_xx": u_xx}


# Number of iterations
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(
    HarmonicOscillator,
    num_inputs=1,
    num_outputs=1,
    num_hidden_layers=2,
    num_neurons_per_layer=25,
)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, N, 0.001)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N)

x = tf.linspace(0.0, 6.283, 600, axis=0)
y = solver.model(x)

plt.plot(x, np.sin(x), "b")
plt.plot(x, y, "r--")
plt.scatter(tf.linspace(0.0, 6.283, 20), [0] * 20, s=5, c="k")
plt.grid()
plt.xlim(0, 6.283)
plt.plot([0, 7], [0, 0], lw=1, c="k")
plt.show()
