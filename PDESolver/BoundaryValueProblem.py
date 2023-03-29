from PDESolver.Sampling import *
import tensorflow as tf


class Condition:
    def __init__(self, name, residue_fn, region_samples_pair, sampler=Random()):
        """
        Class for defining conditions for boundary value problems.

        Parameters
        -----------
        name: str
            Name of the condition
        residue_fn: dict ⟼ tensor
            Function that calculates the residue of the condition
        region_samples_pair: tuple
            Tuple of (region, number of samples)
        sampler: Sampler
            Sampler to use for sampling points in the region
        weight: float
            Weight of the condition
        """

        self.name = name
        self.residue_fn = residue_fn
        self.sample_points = lambda: region_samples_pair[0].pick(
            region_samples_pair[1], sampler)
        self._region = region_samples_pair[0]
        
        big_sample = region_samples_pair[0].pick(int(1e5), sampler)
        self._mean = tf.math.reduce_mean(big_sample, 0)
        self._variance = tf.math.reduce_variance(big_sample, 0)

    def get_region_bounds(self):
        """
        Returns the bounds of the region.

        Returns
        -----------
        tuple: Tuple of (lower bound, upper bound)
        """

        return self._region.get_bounds()
    
    def get_normalization_constants(self):
        """
        Returns the normalization constants of the region.

        Returns
        -----------
        tuple: Tuple of (mean, variance)
        """
        return self._mean, self._variance


class BoundaryValueProblem:
    """
    Class for defining boundary value problems.
    """

    def __init__(self):
        """
        Constructor for a BoundaryValueProblem.
        """

        ...

    def get_conditions(self):
        """
        Returns the conditions of the boundary value problem.

        Returns
        -----------
        list: List of conditions
        """

        return []

    def calculate_differentials(self, model, freeVariables):
        """
        Calculates the differentials of the model at the given points.

        Parameters
        -----------
        model: neural network
            Model to calculate the differentials of
        freeVariables: tensor
            Tensor of points to calculate the differentials at

        Returns
        -----------
        tensor: Tensor of differentials
        """
    
        ...

    def get_debug_string(self):
        debug = ""
        for cond in self.get_conditions():
            samples = len(cond.sample_points())
            debug += "%s: %d\t" % (cond.name, samples)

        return debug[:-1]


class Laplace(BoundaryValueProblem):
    """
    Class defining the Laplace equation in 2D as a boundary value problem.

    (x, y) ⟼ z
    """

    def __init__(self):
        """
        Constructor for a Laplace equation.
        """

        super().__init__()
        self.zero_boundary = Union(Cuboid([0, 0], [0, 1]), Cuboid([0, 0], [1, 0]), Cuboid([1, 0], [1, 1]))
        self.f_boundary = Cuboid([0, 1], [1, 1])
        self.inner = Cuboid([0, 0], [1, 1])

    def get_conditions(self):
        return [
            Condition("zero_boundary",
                      lambda Du: Du["u"],
                      (self.zero_boundary, 180)),
            Condition("f_boundary",
                      lambda Du: Du["u"] - tf.where(tf.abs(Du["x"] - 0.5) < 0.25, 1., 0.), #2 * Du["x"] * (1 - Du["x"]),
                      (self.f_boundary, 400)),
            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["u_yy"],
                      (self.inner, 1600))
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

            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

        del tape

        return {"x": x, "y": y, "u": u, "u_x": u_x, "u_y": u_y, "u_xx": u_xx, "u_yy": u_yy}


class WaveEquation1D(BoundaryValueProblem):
    """
    Class defining the 1D wave equation as a boundary value problem.

    (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a 1D wave equation.
        """

        super().__init__()
        self.initial = Cuboid([0, -1], [0, 1])
        self.boundary1 = Cuboid([0, -1], [0, 1])
        self.inner = Cuboid([0, -1], [1, 1])

    
    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 1 / (200*(Du["x"] + 0.5)**2 + 1) - 1 / (200*(Du["x"] - 0.5)**2 + 2),
                      (self.initial, 128)),
            Condition("boundary1",
                      lambda Du: Du["u_t"],
                      (self.boundary1, 128)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"],
                      (self.inner, 128))
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

    (t, x, y) ⟼ z
    """

    def __init__(self):
        """
        Constructor for a 2D wave equation.
        """

        super().__init__()
        self.initial_u = Cuboid([0, -2, -2], [0, 2, 2])
        self.inner = Cuboid([0, -2, -2], [2, 2, 2])

    def get_conditions(self):
        return [
            Condition("initial_u",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 4,
                      (self.initial_u, 400)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"] - Du["u_yy"],
                      (self.inner, 4900))
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


class HeatEquation1D(BoundaryValueProblem):
    """
    Class defining the 1D heat equation as a boundary value problem.

    (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a 1D heat equation.
        """

        super().__init__()
        self.initial = Cuboid([0, 0], [0, 1])
        self.boundary = Union(Cuboid([0, 0], [1, 0]), Cuboid([0, 1], [1, 1]))
        self.inner = Cuboid([0, 0], [1, 1])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 8*Du["x"]**2 * (1 - Du["x"])**2,
                      (self.initial, 100)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (self.boundary, 100)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.125 * Du["u_xx"],
                      (self.inner, 1600),
                      Equidistant())
        ]

    @staticmethod
    def calculate_differentials(model, freeVariables):
        with tf.GradientTape(persistent=True) as tape:
            t, x = freeVariables[:, 0:1], freeVariables[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            ipt = tf.stack([t[:, 0], x[:, 0]], axis=1)
            u = model(ipt)

            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)

            u_xx = tape.gradient(u_x, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_xx": u_xx}


class HeatEquation2D(BoundaryValueProblem):
    """
    Class defining the 2D heat equation as a boundary value problem.

    (t, x, y) ⟼ z
    """

    def __init__(self):
        """
        Constructor for a 2D heat equation.
        """

        super().__init__()
        self.initial = Cuboid([0, -2, -2], [0, 2, 2])
        self.inner = Cuboid([0, -2, -2], [2, 2, 2])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 2,
                      (self.initial, 400)),
            Condition("inner",
                      lambda Du: Du["u_t"] - Du["u_xx"] - Du["u_yy"],
                      (self.inner, 3600))
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

    (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a Burgers equation.
        """

        super().__init__()
        self.initial = Cuboid([0, -1], [0, 1])
        self.boundary = Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1]))
        self.inner = Cuboid([0, -1], [1, 1])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] + tf.sin(np.pi * Du["x"]),
                      (self.initial, 200)),
            Condition("boundary",
                      lambda Du: Du["u"],
                      (self.boundary, 200)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u"] * Du["u_x"] - 0.01 / np.pi * Du["u_xx"],
                      (self.inner, 1600))
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

    (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a Van-der-Pol equation.
        """

        super().__init__()
        self.initial = Cuboid([0, 0], [4, 0])
        self.boundary = Cuboid([0, 0], [4, 0])
        self.helper = Cuboid([0, 0], [0, 10])
        self.inner = Cuboid([0, 0], [4, 10])

    def conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - 1,
                      (self.initial, 200)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (self.boundary, 200)),
            Condition("helper",
                      lambda Du: Du["u"] - tf.cos(Du["x"]),
                      (self.helper, 200)),
            Condition("inner",
                      lambda Du: Du["u_xx"] - Du["t"] * (1 - Du["u"] ** 2) * Du["u_x"] + Du["u"],
                      (self.inner, 3000))
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

    (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a Allen-Cahn equation.
        """

        super().__init__()
        self.initial = Cuboid([0, -1], [0, 1])
        self.boundary = Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1]))
        self.center = Cuboid([0, 0], [1, 0])
        self.inner = Cuboid([0, -1], [1, 1])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - Du["x"] ** 2 * tf.cos(np.pi * Du["x"]),
                      (self.initial, 100)),
            Condition("boundary1",
                      lambda Du: Du["u"] + 1,
                      (self.boundary, 200)),
            Condition("boundary2",
                      lambda Du: Du["u_x"],
                      (self.boundary, 200)),
            Condition("center",
                      lambda Du: Du["u"],
                      (self.center, 100)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.0001 * Du["u_xx"] + 5 * (Du["u"] ** 3 - Du["u"]),
                      (self.inner, 2500))
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


class KortewegDeVriesEquation(BoundaryValueProblem):
    """
        Class defining the KortewegDeVriesEquation equation.

        (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a KortewegDeVriesEquation equation.
        """

        super().__init__()
        self.initial = Cuboid([0, -1], [0, 1])
        self.inner = Cuboid([0, -1], [1, 1])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - tf.cos(np.pi * Du["x"]),
                      (self.initial, 500)),
            Condition("inner",
                      lambda Du: Du["u_t"] + 6 * Du["u"] * Du["u_x"] + Du["u_xxx"],
                      (self.inner, 2000))
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

            u_xxx = tape.gradient(u_xx, x)

        del tape

        return {"t": t, "x": x, "u": u, "u_t": u_t, "u_x": u_x, "u_xxx": u_xxx}


class ConvectionDiffusionEquation(BoundaryValueProblem):
    """
    Class defining the Convection-Diffusion equation as a boundary value problem.

    (t, x) ⟼ y
    """

    def __init__(self):
        """
        Constructor for a Convection-Diffusion equation.
        """

        super().__init__()
        self.initital = Cuboid([0, -0.5], [0, 1])
        self.inner = Cuboid([0, -0.5], [1, 1])

    def get_conditions(self):
        return [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-25 * Du["x"]**2),
                      (self.initital, 100)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u_x"] - 0.1 * Du["u_xx"],
                      (self.inner, 1000))
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

    (x, y) ⟼ z
    """

    def __init__(self):
        """
        Constructor for a minimal surface equation.
        """

        super().__init__()
        self.boundary1 = Cuboid([0, -1], [0, 1])
        self.boundary2 = Cuboid([1, -1], [1, 1])
        self.inner = Cuboid([0, -1], [1, 1])

    def get_conditions(self):
        return [
            Condition("boundary1",
                      lambda Du: Du["u"] - Du["x"],
                      (self.boundary1, 50)),
            Condition("boundary2",
                      lambda Du: Du["u"] + Du["x"],
                      (self.boundary2, 50)),
            Condition("inner",
                      lambda Du: (1 + Du["u_x"] ** 2) * Du["u_yy"] - 2 * Du["u_y"] * Du["u_x"] * Du["u_xy"] + (
                              1 + Du["u_y"] ** 2) * Du["u_xx"],
                      (self.inner, 500))
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


class Pendulum(BoundaryValueProblem):
    """
    Class defining the differential algebraic equation for a pendulum.

    t ⟼ (x_1, x_2, lambda)
    """

    def __init__(self):
        """
        Constructor for a pendulum.
        """

        super().__init__()
        self.initialPos = Cuboid([0], [0])
        self.initialVel = Cuboid([0], [0])
        self.inner = Cuboid([0], [10])
        self.constraint = Cuboid([0], [10])

    def get_conditions(self):
        def initPos(t):
            return tf.concat([0 * t + 1, 0 * t], axis=1)
        
        def initVel(t):
            return tf.concat([0 * t + 0, 0 * t + 0], axis=1)

        def ode(u, t, alpha):
            return -alpha * u - tf.concat([0 * t, 0 * t + 1.5], axis=1)

        return [
            Condition("initialPos",
                      lambda Du: Du["u"] - initPos(Du["t"]),
                      (self.initialPos, 128)),
            Condition("initialVel",
                      lambda Du: Du["u_t"] - initVel(Du["t"]),
                      (self.initialVel, 128)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - ode(Du["u"], Du["t"], Du["alpha"]),
                      (self.inner, 128)),
            Condition("constraint",
                      lambda Du: tf.norm(Du["u"], axis=1)**2 - 1.,
                      (self.constraint, 128))
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
