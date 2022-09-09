from PDESolver import *


class Oscillator(BoundaryValueProblem):
    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 1,
                      (Cuboid([1, 0], [2, 0]), 50)),
            Condition("initial_x",
                      lambda Du: Du["u_x"],
                      (Cuboid([1, 0], [2, 0]), 50)),
            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["t"] * Du["u"],
                      (Cuboid([1, 0], [2, 6.283]), 2500))
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
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(Oscillator, num_inputs=2, num_outputs=1, num_hidden_layers=4, num_neurons_per_layer=16)
lr = tf.keras.optimizers.schedules.PolynomialDecay(0.01, N, 1e-5, 2)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N)


N = 1000
tspace = np.linspace(1, 2, N + 1)
xspace = np.linspace(0, 6.283, N + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(), X.flatten()]).T

upred = solver.model(tf.cast(Xgrid, 'float32'))[:, 0]
minu, maxu = np.min(upred), np.max(upred)

U = upred.numpy().reshape(N + 1, N + 1)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, U, cmap='viridis')

for t in np.linspace(1, 2, 10):
    x = np.linspace(0, 6.283, 500)
    y = [t] * 500
    z = np.cos(t * x)
    ax.plot(y, x, z, c='r')

ax.view_init(35, 35)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u(t, x)$')
ax.set_title('Solution of equation')
plt.show()
