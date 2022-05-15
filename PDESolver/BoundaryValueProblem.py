from PDESolver.Sampling import *


class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Condition:
    def __init__(self, name, residue_fn, region_samples_pair, sampler=Random(), weight=1):
        """
        Class for defining conditions for boundary value problems.

        :param name: Name of the condition
        :param residue_fn: Function that calculates the residue of the condition
        :param region_samples_pair: Tuple of (region, number of samples)
        :param sampler: Sampler to use for sampling points in the region
        :param weight: Weight of the condition
        """

        self.name = name
        self.residue_fn = residue_fn
        self.sample_points = lambda: region_samples_pair[0].pick(region_samples_pair[1], sampler)
        self._region = region_samples_pair[0]
        self.weight = weight

    def __call__(self, model, bvp):
        Du = bvp.calculate_differentials(model, self.sample_points())
        return self.residue_fn(Du)

    def get_region_bounds(self):
        """
        Returns the bounds of the region.

        :return: Tuple of (lower bound, upper bound)
        """

        return self._region.get_bounds()


class BoundaryValueProblem:
    """
    Class for defining boundary value problems.
    """

    @classproperty
    def conditions(cls):
        """
        Returns the conditions of the boundary value problem.
        :return: List of conditions
        """

        return []

    @staticmethod
    def calculate_differentials(model, freeVariables):
        """
        Calculates the differentials of the model at the given points.

        :param model: Model to calculate the differentials of
        :param freeVariables: Tensor of points to calculate the differentials at
        :return: Tensor of differentials
        """
        ...


class WaveEquation1D(BoundaryValueProblem):
    """
    Class defining the 1D wave equation as a boundary value problem.

    (t, x) |-> y
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - tf.cos(np.pi / 2 * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), 50)),
            Condition("boundary1",
                      lambda Du: Du["u_t"],
                      (Cuboid([0, -1], [0, 1]), 50)),
            Condition("boundary2",
                      lambda Du: Du["u"],
                      (Union(Cuboid([0, -1], [2, -1]), Cuboid([0, 1], [2, 1])), 100)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"],
                      (Cuboid([0, -1], [2, 1]), 1000))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_tt = tape.gradient(u_t, t)
            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_tt": u_tt, "u_xx": u_xx}


class WaveEquation2D(BoundaryValueProblem):
    """
    Class defining the 2D wave equation as a boundary value problem.

    (t, x, y) |-> z
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial_u",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 4,
                      (Cuboid([0, -2, -2], [0, 2, 2]), 400)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, -2, -2], [2, 2, 2]), 4900))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x, y = freeVariables[:, 0:1], freeVariables[:, 1:2], freeVariables[:, 2:3]

            tape.watch(t)
            tape.watch(x)
            tape.watch(y)

            u = model(tf.stack([t[:, 0], x[:, 0], y[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)

            u_tt = tape.gradient(u_t, t)
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

        del tape

        return {"t": t, "x": x, "y": y, "u": u, "u_t": u_t, "u_x": u_x, "u_y": u_yy, "u_tt": u_tt, "u_xx": u_xx,
                "u_yy": u_yy}


class HeatEquation2D(BoundaryValueProblem):
    """
    Class defining the 2D heat equation as a boundary value problem.

    (t, x, y) |-> z
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 2,
                      (Cuboid([0, -2, -2], [0, 2, 2]), 400)),
            Condition("inner",
                      lambda Du: Du["u_t"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, -2, -2], [2, 2, 2]), 3600))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x, y = freeVariables[:, 0:1], freeVariables[:, 1:2], freeVariables[:, 2:3]

            tape.watch(t)
            tape.watch(x)
            tape.watch(y)

            u = model(tf.stack([t[:, 0], x[:, 0], y[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)

            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

        del tape

        return {"t": t, "x": x, "y": y, "u": u, "u_t": u_t, "u_x": u_x, "u_y": u_yy, "u_xx": u_xx, "u_yy": u_yy}


class ControlledHeatEquation1D(BoundaryValueProblem):
    """
    Class defining the 1D heat equation as a boundary value problem with an aspect of control. (EXPERIMENTAL)

    (t, x) |-> (y, control)
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"],
                      (Cuboid([0, 0], [0, 1]), 100)),
            Condition("control",
                      lambda Du: Du["u"] - Du["uR"],
                      (Cuboid([0, 1], [1, 1]), 100)),
            Condition("goal",
                      lambda Du: Du["u"] - (1 - tf.cos(np.pi * Du["x"])) / 2,
                      (Cuboid([1, 0], [1, 1]), 100)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (Union(Cuboid([0, 0], [1, 0]), Cuboid([0, 1], [1, 1])), 200)),
            Condition("inner",
                      lambda Du: Du["u_t"] - Du["u_xx"],
                      (Cuboid([0, 0], [1, 1]), 1600))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        def y(x):
            return model(x)[:, 0]

        def control(x):
            return model(x)[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            ipt = tf.stack([t[:, 0], x[:, 0]], axis=1)
            u = y(ipt)
            uR = control(ipt)

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_xx": u_xx, "uR": uR}


class ControlledHeatEquation2D(BoundaryValueProblem):
    """
    Class defining the 2D heat equation as a temporal boundary value problem. (EXPERIMENTAL)

    (t, x, y) |-> z
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 0.5,
                      (Cuboid([0, 0, 0], [0, 1, 1]), 400),
                      weight=2),
            Condition("boundary",
                      lambda Du: Du["u"] - 1,
                      (Cuboid([5, 1, 0], [5, 1, 1]), 400),
                      weight=2),
            Condition("inner",
                      lambda Du: Du["u_t"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, 0, 0], [5, 1, 1]), 1600))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x, y = freeVariables[:, 0:1], freeVariables[:, 1:2], freeVariables[:, 2:3]

            tape.watch(t)
            tape.watch(x)
            tape.watch(y)

            u = model(tf.stack([t[:, 0], x[:, 0], y[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)

            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

        del tape

        return {"t": t, "x": x, "y": y, "u": u, "u_t": u_t, "u_x": u_x, "u_y": u_yy, "u_xx": u_xx, "u_yy": u_yy}


class BurgersEquation(BoundaryValueProblem):
    """
    Class defining the Burgers equation as a boundary value problem.

    (t, x) |-> y
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] + tf.sin(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), 200)),
            Condition("boundary",
                      lambda Du: Du["u"],
                      (Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1])), 200)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u"] * Du["u_x"] - 0.01 / np.pi * Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), 2000))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_xx": u_xx}


class VanDerPolEquation(BoundaryValueProblem):
    """
    Class defining the Van-der-Pol equation as a boundary value problem.

    (t, x) |-> y
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 1,
                      (Cuboid([0, 0], [4, 0]), 200)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (Cuboid([0, 0], [4, 0]), 200)),
            Condition("helper",
                      lambda Du: Du["u"] - tf.cos(Du["x"]),
                      (Cuboid([0, 0], [0, 10]), 200)),
            Condition("inner",
                      lambda Du: Du["u_xx"] - Du["t"] * (1 - Du["u"] ** 2) * Du["u_x"] + Du["u"],
                      (Cuboid([0, 0], [4, 10]), 3000))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

            u_x = tape.gradient(u, x)

            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_x": u_x, "u_xx": u_xx}


class AllenCahnEquation(BoundaryValueProblem):
    """
    Class defining the Allen-Cahn equation as a boundary value problem.

    (t, x) |-> y
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - Du["x"] ** 2 * tf.cos(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), 100),
                      weight=50),
            Condition("boundary1",
                      lambda Du: Du["u"] + 1,
                      (Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1])), 200)),
            Condition("boundary2",
                      lambda Du: Du["u_x"],
                      (Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1])), 200)),
            Condition("center",
                      lambda Du: Du["u"],
                      (Cuboid([0, 0], [1, 0]), 100),
                      weight=10),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.0001 * Du["u_xx"] + 5 * (Du["u"] ** 3 - Du["u"]),
                      (Cuboid([0, -1], [1, 1]), 2500))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_xx": u_xx}


class ReactionDiffusionEquation(BoundaryValueProblem):
    """
    Class defining the Reaction-Diffusion equation as a boundary value problem.

    (t, x) |-> y
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - tf.cos(np.pi / 2 * Du["x"]) ** 2,
                      (Cuboid([0, -1], [0, 1]), 200),
                      weight=5),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (Union(Cuboid([0, -1], [1.5, -1]), Cuboid([0, 1], [1.5, 1])), 200)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.01 * Du["u_xx"] - 10 * (1 - Du["u"]) * Du["u"] * (Du["u"] - .4),
                      (Cuboid([0, -1], [1.5, 1]), 3600))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_xx": u_xx}


class MinimalSurfaceEquation(BoundaryValueProblem):
    """
    Class defining the minimal surface equation as a boundary value problem.

    (x, y) |-> z
    """

    @classproperty
    def conditions(cls):
        return [
            Condition("boundary1",
                      lambda Du: Du["u"] - Du["x"],
                      (Cuboid([0, -1], [0, 1]), 50)),
            Condition("boundary2",
                      lambda Du: Du["u"] + Du["x"],
                      (Cuboid([1, -1], [1, 1]), 50)),
            Condition("inner",
                      lambda Du: (1 + Du["u_x"] ** 2) * Du["u_yy"] - 2 * Du["u_y"] * Du["u_x"] * Du["u_xy"] + (
                              1 + Du["u_y"] ** 2) * Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), 500))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            x, y = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(x)
            tape.watch(y)

            u = model(tf.stack([x[:, 0], y[:, 0]], axis=1))

            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)

            u_xy = tape.gradient(u_x, y)

            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

        del tape

        return {"x": x, "y": y, "u": u, "u_x": u_x, "u_y": u_y, "u_xy": u_xy, "u_xx": u_xx, "u_yy": u_yy}


class FluidEquation2D(BoundaryValueProblem):
    """
    Class defining the fluid equation as a boundary value problem. (EXPERIMENTAL)

    (t, x, y) |-> (v_x, v_y)
    """

    @classproperty
    def conditions(cls):
        def field(x, y):
            return tf.concat([(1 - x) * x * (1 - y) * y, 0 * x], axis=1)

        def velocity(u, u_x, u_y):
            return -tf.stack([u[:, 0] * u_x[:, 0] + u[:, 1] * u_y[:, 0], u[:, 0] * u_x[:, 1] + u[:, 1] * u_y[:, 1]],
                             axis=1)

        return [
            Condition("initial",
                      lambda Du: Du["u"] - field(Du["x"], Du["y"]),
                      (Cuboid([0, 0, 0], [0, 1, 1]), 400)),
            Condition("inner",
                      lambda Du: Du["u_t"] - velocity(Du["u"], Du["u_x"], Du["u_y"]),
                      (Cuboid([0, 0, 0], [1, 1, 1]), 900)),
            Condition("inner3",
                      lambda Du: Du["ux_x"] + Du["uy_y"],
                      (Cuboid([0, 0, 0], [1, 1, 1]), 900)),
            Condition("boundary",
                      lambda Du: Du["u"],
                      (Union(Cuboid([0, 0, 0], [1, 1, 0]), Cuboid([0, 0, 0], [1, 0, 1]),
                             Cuboid([0, 1, 0], [1, 1, 1]), Cuboid([0, 0, 1], [1, 1, 1])), 900))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        def u1(x):
            return model(x)[:, 0]

        def u2(x):
            return model(x)[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            t, x, y = freeVariables[:, 0:1], freeVariables[:, 1:2], freeVariables[:, 2:3]

            tape.watch(t)
            tape.watch(x)
            tape.watch(y)

            ipt = tf.stack([t[:, 0], x[:, 0], y[:, 0]], axis=1)
            ux = u1(ipt)
            uy = u2(ipt)

            ux_t = tape.gradient(ux, t)
            ux_x = tape.gradient(ux, x)
            ux_y = tape.gradient(ux, y)

            uy_t = tape.gradient(uy, t)
            uy_x = tape.gradient(uy, x)
            uy_y = tape.gradient(uy, y)

            u = tf.stack([ux, uy], axis=1)
            u_t = tf.concat([ux_t, uy_t], axis=1)
            u_x = tf.concat([ux_x, uy_x], axis=1)
            u_y = tf.concat([ux_y, uy_y], axis=1)

        del tape

        return {"t": t, "x": x, "y": y, "u": u, "ux": ux, "uy": uy, "ux_t": ux_t, "ux_x": ux_x, "ux_y": ux_y,
                "uy_t": uy_t,
                "uy_x": uy_x, "uy_y": uy_y, "u_t": u_t, "u_x": u_x, "u_y": u_y}


class VectorTest(BoundaryValueProblem):
    """
    Class defining a equation for reconstructing a gradient field. (EXPERIMENTAL)

    (t, x, y) |-> (v_x, v_y)
    """
    @classproperty
    def conditions(cls):
        def field(x, y):
            return -0.25 * tf.exp(-x ** 2 - y ** 2) * tf.concat([x, y], axis=1)

        return [
            Condition("initial",
                      lambda Du: Du["u"] - field(Du["x"], Du["y"]),
                      (Cuboid([0, -2, -2], [0, 2, 2]), 1600)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u"] / (1 - Du["t"]),
                      (Cuboid([0, -2, -2], [2, 2, 2]), 2700))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        def u1(x):
            return model(x)[:, 0]

        def u2(x):
            return model(x)[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            t, x, y = freeVariables[:, 0:1], freeVariables[:, 1:2], freeVariables[:, 2:3]

            tape.watch(t)
            tape.watch(x)
            tape.watch(y)

            ipt = tf.stack([t[:, 0], x[:, 0], y[:, 0]], axis=1)
            ux = u1(ipt)
            uy = u2(ipt)

            ux_t = tape.gradient(ux, t)

            uy_t = tape.gradient(uy, t)

            u = tf.stack([ux, uy], axis=1)
            u_t = tf.concat([ux_t, uy_t], axis=1)

        del tape

        return {"t": t, "x": x, "y": y, "u": u, "u_t": u_t}


class DAE(BoundaryValueProblem):
    """
    Class defining the differential algebraic equation for a pendulum.

    t |-> (x_1, x_2, lambda)
    """

    @classproperty
    def conditions(cls):
        def initPos(t):
            return tf.concat([0 * t + 1, 0 * t], axis=1)

        def ode(u, t, alpha):
            return -alpha * u - tf.concat([0 * t, 0 * t + 1.5], axis=1)

        return [
            Condition("initialPos",
                      lambda Du: Du["u"] - initPos(Du["t"]),
                      (Cuboid([0], [0]), 10)),
            Condition("initialVel",
                      lambda Du: Du["u_t"],
                      (Cuboid([0], [0]), 10)),
            Condition("ode",
                      lambda Du: Du["u_tt"] - ode(Du["u"], Du["t"], Du["alpha"]),
                      (Cuboid([0], [10]), 500)),
            Condition("constraint",
                      lambda Du: tf.norm(Du["u"], axis=1)**2 - 1.,
                      (Cuboid([0], [10]), 500))
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        def u1(x):
            return model(x)[:, 0]

        def u2(x):
            return model(x)[:, 1]

        def lagrange(x):
            return model(x)[:, 2]


        with tf.GradientTape(persistent=True) as tape:
            t = freeVariables[:, 0:1]

            tape.watch(t)

            ux = u1(t)
            uy = u2(t)
            alpha = tf.transpose(tf.stack([lagrange(t)]))

            ux_t = tape.gradient(ux, t)
            uy_t = tape.gradient(uy, t)

            ux_tt = tape.gradient(ux_t, t)
            uy_tt = tape.gradient(uy_t, t)

            u = tf.stack([ux, uy], axis=1)
            u_t = tf.concat([ux_t, uy_t], axis=1)
            u_tt = tf.concat([ux_tt, uy_tt], axis=1)

        del tape

        return {"t": t, "u": u, "u_t": u_t, "u_tt": u_tt, "alpha": alpha}
