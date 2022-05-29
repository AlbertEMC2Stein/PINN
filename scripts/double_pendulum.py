from PDESolver import *


class WIP_DoublePendulum(BoundaryValueProblem):
    """
    Class defining the differential algebraic equation for a double pendulum.

    t ⟼ (x_1, y_1, x_2, y_2, lambda_1, lambda_2)
    """

    @classproperty
    def conditions(cls):
        g = 1.5

        def initPos(t):
            return tf.concat([0*t + 1, 0*t, 0*t + 1, 0*t - 1], axis=1)

        def ode(t, u, alpha1, alpha2):
            l1 = tf.concat([alpha1 + alpha2, 0*t, -alpha2, 0*t], axis=1)
            l2 = tf.concat([0*t, alpha1 + alpha2, 0*t, -alpha2], axis=1)
            l3 = tf.concat([-alpha2, 0*t, alpha2, 0*t], axis=1)
            l4 = tf.concat([0*t, -alpha2, 0*t, alpha2], axis=1)
            l = tf.stack([l1, l2, l3, l4], axis=1)
            M = tf.reshape(l @ tf.transpose(tf.stack([tf.transpose(u)])), u.shape)
            return M - tf.concat([0*t, 0*t + g, 0*t, 0*t + g], axis=1)

        return [
            Condition("initialPos",
                      lambda Du: Du["u"] - initPos(Du["t"]),
                      (Cuboid([0], [0]), 10)),
            Condition("initialVel",
                      lambda Du: Du["u_t"],
                      (Cuboid([0], [0]), 10)),
            Condition("ode",
                      lambda Du: Du["u_tt"] - ode(Du["t"], Du["u"], Du["alpha1"], Du["alpha2"]),
                      (Cuboid([0], [10]), 500)),
            Condition("constraint1",
                      lambda Du: tf.sqrt(Du["ux1"]**2 + Du["uy1"]**2) - 1.,
                      (Cuboid([0], [10]), 500)),
            Condition("constraint2",
                      lambda Du: tf.sqrt((Du["ux2"] - Du["ux1"])**2 + (Du["uy2"] - Du["uy1"])**2) - 1.,
                      (Cuboid([0], [10]), 500))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        def u11(x):
            return model(x)[:, 0]

        def u12(x):
            return model(x)[:, 1]

        def u21(x):
            return model(x)[:, 2]

        def u22(x):
            return model(x)[:, 3]

        def lagrange1(x):
            return model(x)[:, 4]

        def lagrange2(x):
            return model(x)[:, 5]

        with tf.GradientTape(persistent=True) as tape:
            t = freeVariables[:, 0:1]

            tape.watch(t)

            ux1 = u11(t)
            uy1 = u12(t)
            ux2 = u21(t)
            uy2 = u22(t)
            alpha1 = tf.transpose(tf.stack([lagrange1(t)]))
            alpha2 = tf.transpose(tf.stack([lagrange2(t)]))

            ux1_t = tape.gradient(ux1, t)
            uy1_t = tape.gradient(uy1, t)
            ux2_t = tape.gradient(ux2, t)
            uy2_t = tape.gradient(uy2, t)

            ux1_tt = tape.gradient(ux1_t, t)
            uy1_tt = tape.gradient(uy1_t, t)
            ux2_tt = tape.gradient(ux2_t, t)
            uy2_tt = tape.gradient(uy2_t, t)

            u = tf.stack([ux1, uy1, ux2, uy2], axis=1)
            u_t = tf.concat([ux1_t, uy1_t, ux2_t, uy2_t], axis=1)
            u_tt = tf.concat([ux1_tt, uy1_tt, ux2_tt, uy2_tt], axis=1)

        del tape

        return {"t": t, "ux1": ux1, "uy1": uy1, "ux2": ux2, "uy2": uy2, "u": u, "u_t": u_t, "u_tt": u_tt, "alpha1": alpha1, "alpha2": alpha2}


# Number of iterations
N = 1000

# Initialize solver, learning rate scheduler and choose optimizer
solver = Solver(WIP_DoublePendulum, num_inputs=1, num_outputs=6)
lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-9, N, 1e-3, 2)
optim = tf.keras.optimizers.SGD(learning_rate=lr)

# Train model and plot results
solver.train(optim, lr, N)

fig, ax = plt.subplots()

tspace = tf.linspace([0], [10], 600, axis=0)
xy = solver.model(tspace)
x1d, y1d, x2d, y2d = xy[:, 0].numpy(), xy[:, 1].numpy(), xy[:, 2].numpy(), xy[:, 3].numpy()
pendulum1, = plt.plot([], [], 'lightgray')
pendulum2, = plt.plot([], [], 'lightgray')
ln, = plt.plot([], [], 'cornflowerblue')
sc1, = plt.plot([], [], 'bo', markersize=10)
sc2, = plt.plot([], [], 'bo', markersize=10)
title = ax.text(0, 0.1, "", ha='center')


def init():
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 0.25)
    return ln, sc1, sc2, pendulum1, pendulum2, title


def update(frame):
    trail = 20
    start = max(frame - trail, 0)
    pendulum1.set_data([0, x1d[frame]], [0, y1d[frame]])
    pendulum2.set_data([x1d[frame], x2d[frame]], [y1d[frame], y2d[frame]])
    ln.set_data(x2d[start:frame + 1], y2d[start:frame + 1])
    sc1.set_data(x1d[frame], y1d[frame])
    sc2.set_data(x2d[frame], y2d[frame])
    title.set_text(u"t = {:.3f}".format(frame))

    return ln, sc1, sc2, pendulum1, pendulum2, title


anim = FuncAnimation(fig, update, frames=np.arange(len(tspace)),
                     init_func=init, blit=True, interval=16)

anim.save("../out/%s_solution.gif" % solver.bvp.__name__, fps=60)
plt.show()
