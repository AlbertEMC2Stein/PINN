import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _outFolderExists():
    if not os.path.isdir('../out/'):
        os.mkdir('../out/')


def _plot_loss(solver):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    n = len(solver.loss_history)
    k = min(100, n)
    averaged_loss = np.convolve(solver.loss_history, np.ones(k) / k, mode='same')

    ax.semilogy(range(n), solver.loss_history, 'k-', lw=0.5)
    ax.semilogy(range(n), averaged_loss, 'r--', lw=1)
    ax.set_xlabel('$Iteration$')
    ax.set_ylabel('$Loss$')
    ax.set_xlim(0, n - (1 if n > 1 else 0))
    plt.show()


def _plot_3d(X, Y, Z, ax, variables, title, show=True):
    ax.view_init(35, 135)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('$%s$' % variables[0])
    ax.set_ylabel('$%s$' % variables[1])
    ax.set_zlabel('$u(%s, %s)$' % (variables[0], variables[1]))
    ax.set_title(title)
    
    plot = ax.plot_surface(X, Y, Z, cmap='jet')
    if show:
        plt.show()

    return plot


def _plot_heatmap(X, Y, Z, ax, variables, title, show=True):
    plot = ax.imshow(np.flip(Z, 0), cmap='jet', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
    ax.set_xlabel('$%s$' % variables[0])
    ax.set_ylabel('$%s$' % variables[1])
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plot, cax=cax)
    if show:
        plt.show()

    return plot
    

def plot_1D(solver):
    """
    Plots the solution of a BVP defined in one spacial dimension and one output.

    Parameters
    -----------
    solver: Solver
        Trained solver to be visualized
    """

    def plot_sample_points():
        fig = plt.figure(figsize=(9, 6))

        for cond in solver.bvp.get_conditions():
            t, x = cond.sample_points()[:, 0], cond.sample_points()[:, 1]

            if cond.name != 'inner':
                plt.scatter(t, x, c='black', marker='X')
            else:
                plt.scatter(t, x, c='r', marker='.', alpha=0.1)

        plt.xlabel('$t$')
        plt.ylabel('$x$')

        plt.title('Positions of collocation points and boundary data')
        plt.show()

    def plot_u():
        # Set up meshgrid
        N = 1000
        xspace = np.linspace(bounds[0][0], bounds[1][0], N + 1)
        yspace = np.linspace(bounds[0][1], bounds[1][1], N + 1)
        X, Y = np.meshgrid(xspace, yspace)
        XYgrid = np.vstack([X.flatten(), Y.flatten()]).T

        # Determine predictions of u(t, x)
        upred = solver.model(tf.cast(XYgrid, 'float32'))[:, 0]
        minu, maxu = np.min(upred), np.max(upred)

        # Reshape upred
        U = upred.numpy().reshape(N + 1, N + 1)

        # Surface plot of solution u(t,x)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        _plot_3d(X, Y, U, ax, ['t', 'x'], 'Solution of equation', False)
        plt.savefig('../out/3D_%s_solution.pdf' % solver.bvp.__class__.__name__, bbox_inches='tight', dpi=300)
        plt.show()

        fig = plt.figure(figsize=(9, 6))
        plt.imshow(U, cmap='viridis')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Solution of equation')

        plt.savefig('../out/2D_%s_solution.pdf' % solver.bvp.__class__.__name__, bbox_inches='tight', dpi=300)
        plt.show()

        return minu, maxu

    def animate_solution():
        fig, ax = plt.subplots()
        xdata, ydata = tf.linspace(bounds[0][1], bounds[1][1], 1000), []
        ln, = plt.plot([], [], color='b')
        title = ax.text((bounds[0][1] + bounds[1][1]) / 2, maxu - 0.1 * abs(maxu), "", ha='center')

        def init():
            ax.set_xlim(bounds[0][1], bounds[1][1])
            ax.set_ylim(minu, maxu)

            return ln,

        def update(frame):
            ydata = solver.model(tf.transpose(tf.stack([tf.ones_like(xdata) * frame, xdata])))[:, 0]
            ln.set_data(xdata, ydata)
            title.set_text(u"t = {:.3f}".format(frame))

            return ln, title

        anim = FuncAnimation(fig, update, frames=np.linspace(bounds[0][0], bounds[1][0], 300),
                            interval=16, init_func=init, blit=True)

        anim.save("../out/2D_%s_solution.gif" % solver.bvp.__class__.__name__, fps=30)
        plt.show()

    inner = [cond for cond in solver.bvp.conditions if cond.name == "inner"][0]
    bounds = inner.get_region_bounds()

    _outFolderExists()
    plot_sample_points()
    _plot_loss(solver)
    minu, maxu = plot_u()
    animate_solution()


def plot_2D(solver):
    """
    Plots the solution of a BVP defined in two spacial dimensions and one output.

    Parameters
    -----------
    solver: Solver
        Trained solver to be visualized
    """

    def animate_solution():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        N = 100
        xspace = np.linspace(0, 1, N)
        yspace = np.linspace(0, 1, N)
        X, Y = np.meshgrid(xspace, yspace)
        XYgrid = np.vstack([X.flatten(), Y.flatten()]).T
        XYgrid = tf.cast(XYgrid, 'float32')

        # plot = _plot_3d(X, Y, np.zeros_like(X), ax, ['t', 'x'], '', False)

        # def data_gen(frame):
        #     x, y = XYgrid[:, 0], XYgrid[:, 1]
        #     z = solver.model(tf.transpose(tf.stack([tf.ones_like(x) * frame, x, y])))
        #     Z = tf.reshape(z, X.shape)

        #     ax.clear()
        #     plot = ax.plot_surface(X, Y, Z, cmap='viridis')
        #     ax.set_xlim(0, 1)
        #     ax.set_ylim(0, 1)
        #     ax.set_zlim(-0.1, 1)

        #     title = ax.set_title("t = %.2f" % frame, ha='center')
        #     return plot, title
        
        x, y = XYgrid[:, 0], XYgrid[:, 1]
        zarray = np.zeros((N, N, 120))

        for i in range(120):
            z = solver.model(tf.transpose(tf.stack([tf.ones_like(x) * i, x, y])))
            zarray[:, :, i] = tf.reshape(z, X.shape)

        def update_plot(frame_number, zarray, plot):
            plot[0].remove()
            plot[0] = ax.plot_surface(X, Y, zarray[:,:,frame_number], cmap="magma")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plot = [ax.plot_surface(X, Y, zarray[:, :, 0], color='0.75', rstride=1, cstride=1)]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-0.1, 1)
        anim = FuncAnimation(fig, update_plot, 120, fargs=(zarray, plot), interval=16)

        #anim = FuncAnimation(fig, data_gen, fargs=(plot,), frames=np.linspace(0, 2, 120),
        #                     interval=16, blit=False)

        anim.save("../out/%s_solution.gif" % solver.bvp.__class__.__name__, fps=30)
        plt.show()

    _outFolderExists()
    _plot_loss(solver)
    animate_solution()


def plot_2Dvectorfield(solver):
    """
    Plots the solution of a BVP defined in two spacial dimensions and two outputs.

    Parameters
    -----------
    solver: Solver
        Trained solver to be visualized
    """

    def animate_solution():
        fig = plt.figure()
        ax = fig.add_subplot()

        N = 30
        xspace = np.linspace(-2, 2, N)
        yspace = np.linspace(-2, 2, N)
        X, Y = np.meshgrid(xspace, yspace)
        XYgrid = np.vstack([X.flatten(), Y.flatten()]).T
        XYgrid = tf.cast(XYgrid, 'float32')

        x, y = XYgrid[:, 0], XYgrid[:, 1]
        z = solver.model(tf.transpose(tf.stack([tf.zeros_like(x), x, y])))
        Zx = tf.reshape(z[:, 0], X.shape)
        Zy = tf.reshape(z[:, 1], X.shape)
        plot = ax.quiver(X, Y, Zx, Zy, scale=1.0)

        def data_gen(frame):
            x, y = XYgrid[:, 0], XYgrid[:, 1]
            z = solver.model(tf.transpose(tf.stack([tf.ones_like(x) * frame, x, y])))
            Zx = tf.reshape(z[:, 0], X.shape)
            Zy = tf.reshape(z[:, 1], X.shape)

            ax.clear()
            plot = ax.quiver(X, Y, Zx, Zy, scale=1.0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)

            title = ax.set_title("t = %.2f" % frame, ha='center')
            return plot, title

        anim = FuncAnimation(fig, data_gen, fargs=(plot,), frames=np.linspace(0, 2, 240),
                             interval=16, blit=False)

        anim.save("../out/%s_solution.gif" % solver.bvp.__class__.__name__, fps=30)
        plt.show()

    _outFolderExists()
    _plot_loss()
    animate_solution()


def plot_phaseplot(solver):
    """
    Plots the solution of a DAE defined in one temporal dimension and two spacial outputs.

    Parameters
    -----------
    solver: Solver
        Trained solver to be visualized
    """

    def animate_solution():
        fig, ax = plt.subplots(figsize=(5, 5))

        tspace = tf.linspace([0], [10], 600, axis=0)
        xy = solver.model(tspace)
        xd, yd = xy[:, 0].numpy(), xy[:, 1].numpy()
        pendulum, = plt.plot([], [], 'lightgray')
        ln, = plt.plot([], [], 'cornflowerblue')
        sc, = plt.plot([], [], 'bo', markersize=10)
        title = ax.text(0, 1.15, "", ha='center')

        def init():
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-1.25, 1.25)
            return ln, sc, pendulum, title

        def update(frame):
            trail = 20
            start = max(frame - trail, 0)
            pendulum.set_data([0, xd[frame]], [0, yd[frame]])
            ln.set_data(xd[start:frame+1], yd[start:frame+1])
            sc.set_data(xd[frame], yd[frame])
            title.set_text(u"t = {:.3f}".format(frame))

            return ln, sc, pendulum, title

        anim = FuncAnimation(fig, update, frames=np.arange(len(tspace)),
                            init_func=init, blit=True, interval=16)

        #anim.save("../out/%s_solution.gif" % solver.bvp.__class__.__name__, fps=60)
        plt.show()

    _outFolderExists()
    _plot_loss(solver)
    animate_solution()


def debug_plot_2D(solver, variables, domain):
    N = 1000
    xspace = np.linspace(domain[0], domain[1], N + 1)
    yspace = np.linspace(domain[2], domain[3], N + 1)
    X, Y = np.meshgrid(xspace, yspace)
    XYgrid = np.vstack([X.flatten(), Y.flatten()]).T

    upred = solver.model(tf.cast(XYgrid, 'float32'))[:, 0]
    
    U = upred.numpy().reshape(N + 1, N + 1)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    _plot_3d(X, Y, U, ax, variables, 'Solution of equation')

    plt.imshow(np.flip(U, 0), cmap='jet', extent=domain)
    plt.colorbar()
    plt.show()

    _plot_loss(solver)


def error_plot_2D(solver, exact_solution, variables, domain, heatmap=True):
    N = 1000
    xspace = np.linspace(domain[0], domain[1], N + 1)
    yspace = np.linspace(domain[2], domain[3], N + 1)
    X, Y = np.meshgrid(xspace, yspace)
    XYgrid = np.vstack([X.flatten(), Y.flatten()]).T

    upred = solver.model(tf.cast(XYgrid, 'float32'))[:, 0]
    U = upred.numpy().reshape(N + 1, N + 1)

    fig = plt.figure(figsize=(18, 5))
    plt.subplots_adjust(left=0.01, right=0.97)

    options, plot_fun = ({'projection': '3d'}, _plot_3d) if not heatmap else ({}, _plot_heatmap)

    ax = fig.add_subplot(131, **options)
    plot_fun(X, Y, exact_solution(X, Y), ax, variables, 'Exact solution', False)

    ax = fig.add_subplot(132, **options)
    plot_fun(X, Y, U, ax, variables, 'Predicted solution', False)

    ax = fig.add_subplot(133)
    _plot_heatmap(X, Y, np.abs(U - exact_solution(X, Y)), 
                  ax, variables, 'Absolute Error', True)

    _plot_loss(solver)