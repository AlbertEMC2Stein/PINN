import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

matplotlib.use('macosx')

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Condition:
    def __init__(self, name, residue_fn, region_samples_pair, weight=1):
        self.name = name
        self.residue_fn = residue_fn
        self.sample_points = region_samples_pair[0].pick(region_samples_pair[1])
        self._region = region_samples_pair[0]
        self.weight = weight

    def __call__(self, model):
        with tf.GradientTape(persistent=True) as tape:
            t, x = self.sample_points[:, 0:1], self.sample_points[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_tt = tape.gradient(u_t, t)
            u_tx = tape.gradient(u_t, x)
            u_xt = tape.gradient(u_x, t)
            u_xx = tape.gradient(u_x, x)

        del tape

        Du = {"u": u, "u_t": u_t, "u_x": u_x, "u_tt": u_tt, "u_tx": u_tx, "u_xt": u_xt, "u_xx": u_xx}
        return self.residue_fn(t, x, Du)

    def get_region_bounds(self):
        bounds = [np.inf, -np.inf, np.inf, -np.inf]
        for section in self._region.sections:
            for vertex in section:
                if vertex[0] < bounds[0]:
                    bounds[0] = vertex[0].numpy()

                if vertex[0] > bounds[1]:
                    bounds[1] = vertex[0].numpy()

                if vertex[1] < bounds[2]:
                    bounds[2] = vertex[1].numpy()

                if vertex[1] > bounds[3]:
                    bounds[3] = vertex[1].numpy()

        return bounds


class Region:
    def __init__(self, *sections):
        self.sections = [[tf.constant(p, dtype=DTYPE) for p in section] for section in sections]

    def pick(self, n):
        l = len(self.sections)
        ns = [n // l] * l
        ns[-1] += n % l

        points = []
        for i, k in enumerate(ns):
            p = self.sections[i]
            t = tf.random.uniform((k, 1))
            if len(p) == 2:
                points += [(1 - t) * p[0] + t * p[1]]

            elif len(p) == 4:
                s = tf.random.uniform((k, 1))
                points += [(1 - s) * (1 - t) * p[0] + s * (1 - t) * p[1] + (1 - s) * t * p[2] + s * t * p[3]]

        return tf.concat(points, axis=0)


class BVP:
    wave_equation = [
        Condition("initial",
                  lambda t, x, Du: Du["u"] - tf.cos(np.pi / 2 * x) ** 20,
                  (Region([[0, -1], [0, 1]]), 50)),
        Condition("boundary1",
                  lambda t, x, Du: Du["u_t"],
                  (Region([[0, -1], [0, 1]]), 50)),  # [[0, 1], [1, 1]], [[0, -1], [1, -1]]
        Condition("boundary2",
                  lambda t, x, Du: Du["u_x"],
                  (Region([[0, -1], [2, -1]], [[0, 1], [2, 1]]), 100)),
        Condition("inner",
                  lambda t, x, Du: Du["u_tt"] - Du["u_xx"],
                  (Region([[0, -1], [2, -1], [0, 1], [2, 1]]), 1000))
    ]

    burgers_equation = [
        Condition("initial",
                  lambda t, x, Du: Du["u"] + tf.sin(np.pi * x),
                  (Region([[0, -1], [0, 1]]), 50),
                  10),
        Condition("boundary",
                  lambda t, x, Du: Du["u"],
                  (Region([[0, -1], [1, -1]], [[0, 1], [1, 1]]), 100)),
        Condition("inner",
                  lambda t, x, Du: Du["u_t"] + Du["u"] * Du["u_x"] - 0.01/np.pi * Du["u_xx"],
                  (Region([[0, -1], [1, -1], [0, 1], [1, 1]]), 1000))
    ]

    van_der_pol_equation = [
        Condition("initial",
                  lambda t, x, Du: Du["u"] - tf.sin(np.pi * x),
                  (Region([[0, 0], [0, 3]]), 50)),
        Condition("boundary",
                  lambda t, x, Du: Du["u_t"],
                  (Region([[0, 0], [0, 3]]), 50)),
        Condition("inner",
                  lambda t, x, Du: Du["u_tt"] - x * (1 - Du["u"]**2) * Du["u_t"] + Du["u"],
                  (Region([[0, 0], [6, 0]], [[0, 3], [6, 3]]), 1000))
    ]


def init_model(num_hidden_layers=6, num_neurons_per_layer=16):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(2))

    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation=tf.keras.activations.tanh,
                                        kernel_initializer='glorot_normal'))

    model.add(tf.keras.layers.Dense(1))

    return model


def train(iterations):
    def compute_loss(model):
        criterion = tf.keras.losses.Huber()

        loss = 0
        for cond in conditions:
            out = cond(model)
            loss += cond.weight * criterion(out, 0)

        return loss

    def get_grad(model):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(model.trainable_variables)
            loss = compute_loss(model)

        g = tape.gradient(loss, model.trainable_variables)
        del tape

        return loss, g

    @tf.function
    def train_step():
        # Compute current loss and gradient w.r.t. parameters
        loss, grad_theta = get_grad(model)

        # Perform gradient descent step
        optim.apply_gradients(zip(grad_theta, model.trainable_variables))

        return loss

    loss_hist = []
    t0 = time()
    for i in range(iterations + 1):
        # while time() - t0 <= 60:
        loss = train_step()
        loss_hist.append(loss.numpy())

        if i % 50 == 0:
            # if (time() - t0) % 5 < 0.01:
            # print('\rIt {:05d}s: loss = {:10.8e} lr = {:.5f}'.format(int(time() - t0), loss, 0), end="")
            print('\rIt {:05d}: loss = {:10.8e} lr = {:.5f}'.format(i, loss, lr(i)), end="")

    print('\nComputation time: {:.1f} seconds'.format(time() - t0))

    return loss_hist


def plot_results():
    def plot_sample_points():
        fig = plt.figure(figsize=(9, 6))

        for cond in conditions:
            t, x = cond.sample_points[:, 0], cond.sample_points[:, 1]

            if cond.name != 'inner':
                plt.scatter(t, x, c='black', marker='X')
            else:
                plt.scatter(t, x, c='r', marker='.', alpha=0.1)

        plt.xlabel('$t$')
        plt.ylabel('$x$')

        plt.title('Positions of collocation points and boundary data');
        # plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_loss():
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.semilogy(range(len(loss_hist)), loss_hist, 'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi_{n_{epoch}}$');
        plt.show()

    def plot_u():
        # Set up meshgrid
        N = 1000
        tspace = np.linspace(bounds[0], bounds[1], N + 1)
        xspace = np.linspace(bounds[2], bounds[3], N + 1)
        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(), X.flatten()]).T

        # Determine predictions of u(t, x)
        upred = model(tf.cast(Xgrid, DTYPE))
        minu, maxu = np.min(upred), np.max(upred)

        # Reshape upred
        U = upred.numpy().reshape(N + 1, N + 1)

        # Surface plot of solution u(t,x)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, X, U, cmap='viridis')
        ax.view_init(35, 35)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u_\\theta(t,x)$')
        ax.set_title('Solution of equation')
        # plt.savefig('PDE_Solution_3D.pdf', bbox_inches='tight', dpi=300);

        # from ODESolver.One_step_methods import runge_kutta
        #
        # ode = lambda x, c: np.array([x[2], c * (1 - x[1] ** 2) * x[2] - x[1]])
        # for c in np.linspace(0, 3, 31):
        #     print("\rc = %.2f" % c, end="")
        #     x, y = runge_kutta(lambda x: ode(x, c), np.array([1, 0]), [0, 6], 5e-3)
        #     ax.plot(x, np.ones_like(x) * c, y[:, 0], color='red')
        #
        # plt.show()

        fig = plt.figure(figsize=(9, 6))
        plt.imshow(U, cmap='viridis')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Solution of equation')
        # plt.savefig('PDE_Solution_2D.pdf', bbox_inches='tight', dpi=300);
        plt.show()

        return minu, maxu

    def animate_solution():
        fig, ax = plt.subplots()
        xdata, ydata = tf.linspace(bounds[2], bounds[3], 1000), []
        ln, = plt.plot([], [], color='b')
        title = ax.text(0, maxu * 0.9, "", ha='center')

        def init():
            ax.set_xlim(bounds[2], bounds[3])
            ax.set_ylim(minu, maxu)

            return ln,

        def update(frame):
            ydata = model(tf.transpose(tf.stack([tf.ones_like(xdata) * frame, xdata])))
            ln.set_data(xdata, ydata)
            title.set_text(u"t = {:.3f}".format(frame))

            return ln, title

        ani = FuncAnimation(fig, update, frames=np.linspace(bounds[0], bounds[1], 300),
                            interval=16, init_func=init, blit=True)
        plt.show()

    inner = [cond for cond in conditions if cond.name == "inner"][0]
    bounds = inner.get_region_bounds()

    plot_sample_points()
    plot_loss()
    minu, maxu = plot_u()
    animate_solution()


if __name__ == "__main__":
    # Choose BVP
    conditions = BVP.burgers_equation

    # Number of iterations
    N = 10000

    # Initialize model, learning rate scheduler and choose optimizer
    model = init_model()
    lr = tf.keras.optimizers.schedules.PolynomialDecay(0.01, N, 1e-4, 1)
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    loss_hist = train(N)

    plot_results()  # 4.78862858e-05
