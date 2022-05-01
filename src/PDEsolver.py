import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from BoundaryValueProblem import *

matplotlib.use('macosx')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Solver:
    def __init__(self, bvp, num_hidden_layers=6, num_neurons_per_layer=16):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(2))

        for _ in range(num_hidden_layers):
            model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer='glorot_normal'))

        model.add(tf.keras.layers.Dense(1))
        self.model = model
        self.bvp = bvp
        self.loss_history = []

    def train(self, optimizer, lr_scheduler, iterations=10000):
        def compute_loss():
            criterion = tf.keras.losses.Huber()

            loss = 0
            for cond in self.bvp.conditions:
                out = cond(self.model, self.bvp)
                loss += cond.weight * criterion(out, 0)

            return loss

        def get_grad():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.model.trainable_variables)
                loss = compute_loss()

            g = tape.gradient(loss, self.model.trainable_variables)
            del tape

            return loss, g

        @tf.function
        def train_step():
            # Compute current loss and gradient w.r.t. parameters
            loss, grad_theta = get_grad()

            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))

            return loss

        t0 = time()
        for i in range(iterations + 1):
            # while time() - t0 <= 60:
            loss = train_step()
            self.loss_history += [loss.numpy()]

            if i % 50 == 0:
                # if (time() - t0) % 5 < 0.01:
                # print('\rIt {:05d}s: loss = {:10.8e} lr = {:.5f}'.format(int(time() - t0), loss, 0), end="")
                print('\rIt {:05d}: loss = {:10.8e} lr = {:.5f}'.format(i, loss, lr_scheduler(i)), end="")

        print('\nComputation time: {:.1f} seconds'.format(time() - t0))


def plot_results(loss_hist):
    def plot_sample_points():
        fig = plt.figure(figsize=(9, 6))

        for cond in bvp.conditions:
            t, x = cond.sample_points()[:, 0], cond.sample_points()[:, 1]

            if cond.name != 'inner':
                plt.scatter(t, x, c='black', marker='X')
            else:
                plt.scatter(t, x, c='r', marker='.', alpha=0.1)

        plt.xlabel('$t$')
        plt.ylabel('$x$')

        plt.title('Positions of collocation points and boundary data')
        # plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_loss():
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.semilogy(range(len(loss_hist)), loss_hist, 'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi_{n_{epoch}}$')
        plt.show()

    def plot_u():
        # Set up meshgrid
        N = 1000
        tspace = np.linspace(bounds[0][0], bounds[1][0], N + 1)
        xspace = np.linspace(bounds[0][1], bounds[1][1], N + 1)
        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(), X.flatten()]).T

        # Determine predictions of u(t, x)
        upred = solver.model(tf.cast(Xgrid, 'float32'))
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
        xdata, ydata = tf.linspace(bounds[0][1], bounds[1][1], 1000), []
        ln, = plt.plot([], [], color='b')
        title = ax.text(0, maxu * 0.9, "", ha='center')

        def init():
            ax.set_xlim(bounds[0][1], bounds[1][1])
            ax.set_ylim(minu, maxu)

            return ln,

        def update(frame):
            ydata = solver.model(tf.transpose(tf.stack([tf.ones_like(xdata) * frame, xdata])))
            ln.set_data(xdata, ydata)
            title.set_text(u"t = {:.3f}".format(frame))

            return ln, title

        ani = FuncAnimation(fig, update, frames=np.linspace(bounds[0][0], bounds[1][0], 300),
                            interval=16, init_func=init, blit=True)
        plt.show()

    inner = [cond for cond in bvp.conditions if cond.name == "inner"][0]
    bounds = inner.get_region_bounds()

    plot_sample_points()
    plot_loss()
    minu, maxu = plot_u()
    animate_solution()


if __name__ == "__main__":
    bvp = VanDerPolEquation

    # Number of iterations
    N = 10000

    # Initialize model, learning rate scheduler and choose optimizer
    solver = Solver(bvp)
    lr = tf.keras.optimizers.schedules.PolynomialDecay(0.01, N, 1e-4, 2)
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    solver.train(optim, lr, N)
    plot_results(solver.loss_history)
