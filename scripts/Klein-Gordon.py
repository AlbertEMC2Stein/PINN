from PDESolver import *
from numpy import pi

# Parameters of equations
alpha = -1.0
beta = 0.0
gamma = 1.0
k = 3

def u(t, x):
    return x * tf.cos(5 * pi * t) + (x * t)**3

def u_tt(t, x):
    return - 25 * pi**2 * x * tf.cos(5 * pi * t) + 6 * t * x**3

def u_xx(t, x):
    return 6 * x * t**3

def f(t, x, alpha, beta, gamma, k):
    return u_tt(t, x) + alpha * u_xx(t, x) + beta * u(t, x) + gamma * u(t, x)**k

class KleinGordon(BoundaryValueProblem):
    def __init__(self):
        super().__init__()

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - Du["x"],
                      (Cuboid([0, 0], [0, 1]), 128)),
            Condition("initial_t",
                      lambda Du: Du["u_t"],
                      (Cuboid([0, 0], [0, 1]), 128)),
            Condition("LR",
                      lambda Du: Du["u"] - u(Du["t"], Du["x"]),
                      (Cuboid([0, 0], [1, 0]) & Cuboid([0, 1], [1, 1]), 128)),
            Condition("inner",
                      lambda Du: Du["u_tt"] + alpha * Du["u_xx"] + beta * Du["u"] + gamma * Du["u"]**k - f(Du["t"], Du["x"], alpha, beta, gamma, k),
                      (Cuboid([0, 0], [1, 1]), 128))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_tt", "u_xx"])

# Number of iterations
N = 10000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(KleinGordon(), num_hidden_layers=4, num_neurons_per_layer=50)
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9)
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Train model and plot result
solver.train(optim, lr, N, N)

N = 1000
tspace = np.linspace(0, 1, N + 1)
xspace = np.linspace(0, 1, N + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(), X.flatten()]).T

upred = solver.model(tf.cast(Xgrid, 'float32'))[:, 0]
minu, maxu = np.min(upred), np.max(upred)

U = upred.numpy().reshape(N + 1, N + 1)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, U, cmap='jet')

for t in np.linspace(0, 1, 20):
    x = np.linspace(0, 1, 500)
    y = [t] * 500
    z = u(t, x)
    ax.plot(y, x, z, c='r')

ax.view_init(35, 35)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u(t, x)$')
ax.set_title('Solution of equation')
plt.show()

error_plot_2D(solver, u, ['t', 'x'], [0, 1, 0, 1])
