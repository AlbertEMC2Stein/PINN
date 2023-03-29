from PDESolver.Sampling import *
import tensorflow as tf


class Specification:
    def __init__(self, components, variables, differentials):
        """
        Class for defining the specification of a boundary value problem.

        Parameters
        -----------
        components: list
            List of components of the boundary value problem
        variables: list
            List of variables of the boundary value problem
        differentials: list
            List of differentials of the boundary value problem
        """

        self.components = components
        self.variables = variables
        self.differentials = differentials

    def as_dictionary(self):
        """
        Returns the specification as a dictionary.

        Returns
        -----------
        dict: Specification of form {"components": [str], "variables": [str], "differentials": [str]}
        """

        return {"components": self.components, "variables": self.variables, "differentials": self.differentials}


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

    def __init__(self, minibatch_size=128):
        """
        Constructor for a BoundaryValueProblem.
        """

        self.conditions = None
        self.specification = None
        self.minibatch_size = None

    def get_conditions(self):
        """
        Returns the conditions of the boundary value problem.

        Returns
        -----------
        list: List of conditions
        """

        return self.conditions

    def get_specification(self):
        """
        Returns the specification of the boundary value problem.
        Right now, variables are only allowed to be single characters.

        Returns
        -----------
        dict: Specification of form {"components": [str], "variables": [str], "differentials": [str]}
        """

        return self.specification.as_dictionary()


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
 
        self.conditions = [
            Condition("zero_boundary",
                      lambda Du: Du["u"],
                      (Cuboid([0, 0], [0, 1]) & Cuboid([0, 0], [1, 0]) & Cuboid([1, 0], [1, 1]), 128)),

            Condition("f_boundary",
                      # tf.where(tf.abs(Du["x"] - 0.5) < 0.25, 1., 0.),
                      lambda Du: Du["u"] - 2 * Du["x"] * (1 - Du["x"]),
                      (Cuboid([0, 1], [1, 1]), 128)),

            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["u_yy"],
                      (Cuboid([0, 0], [1, 1]), 128))
        ]

        self.specification = Specification(["u"], ["x", "y"], ["u_xx", "u_yy"])


class WaveEquation1D(BoundaryValueProblem):
    """
    Class defining the 1D wave equation as a boundary value problem.

    (t, x) ⟼ y
    """

    def __init__(self, minibatch_size=128):
        """
        Constructor for a 1D wave equation.
        """

        super().__init__()
        
        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-25 * Du["x"]**2), #1 / (200*(Du["x"] + 0.5) ** 2 + 1) - 1 / (200*(Du["x"] - 0.5)**2 + 2),
                      (Cuboid([0, -1], [0, 1]), None)),
            Condition("boundary1",
                      lambda Du: Du["u_t"],
                      (Cuboid([0, -1], [0, 1]), None)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), None))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_tt", "u_xx"])


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

        self.conditions = [
            Condition("initial_u",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 4,
                      (Cuboid([0, -2, -2], [0, 2, 2]), 512)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, -2, -2], [2, 2, 2]), 512))
        ]

        self.specification = Specification(["u"], ["t", "x", "y"], ["u_tt", "u_xx", "u_yy"])


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

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - 8*Du["x"]**2 * (1 - Du["x"])**2,
                      (Cuboid([0, 0], [0, 1]), 100)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (Cuboid([0, 0], [1, 0]) & Cuboid([0, 1], [1, 1]), 100)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.125 * Du["u_xx"],
                      (Cuboid([0, 0], [1, 1]), 1600))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


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

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 2,
                      (Cuboid([0, -2, -2], [0, 2, 2]), 400)),
            Condition("inner",
                      lambda Du: Du["u_t"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, -2, -2], [2, 2, 2]), 3600))
        ]

        self.specification = Specification(["u"], ["t", "x", "y"], ["u_t", "u_xx", "u_yy"])


class BurgersEquation(BoundaryValueProblem):
    """
    Class defining the Burgers equation as a boundary value problem.

    (t, x) ⟼ y
    """

    def __init__(self, viscosity=0.01):
        """
        Constructor for a Burgers equation.
        """

        super().__init__()
        self.viscosity = viscosity

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] + tf.sin(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), 128)),
            Condition("boundary",
                      lambda Du: Du["u"],
                      (Cuboid([0, -1], [1, -1]) & Cuboid([0, 1], [1, 1]), 128)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u"] * Du["u_x"] - self.viscosity / np.pi * Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), 128))
        ]
        
        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


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

        self.conditions = [
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
        
        self.specification = Specification(["u"], ["t", "x"], ["u_xx"])


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

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - Du["x"] ** 2 * tf.cos(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), 100)),
            Condition("boundary1",
                      lambda Du: Du["u"] + 1,
                      (Cuboid([0, -1], [1, -1]) & Cuboid([0, 1], [1, 1]), 200)),
            Condition("boundary2",
                      lambda Du: Du["u_x"],
                      (Cuboid([0, -1], [1, -1]) & Cuboid([0, 1], [1, 1]), 200)),
            Condition("center",
                      lambda Du: Du["u"],
                      (Cuboid([0, 0], [1, 0]), 100)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.0001 * Du["u_xx"] + 5 * (Du["u"] ** 3 - Du["u"]),
                      (Cuboid([0, -1], [1, 1]), 2500))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


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

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.cos(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), 500)),
            Condition("inner",
                      lambda Du: Du["u_t"] + 6 * Du["u"] * Du["u_x"] + Du["u_xxx"],
                      (Cuboid([0, -1], [1, 1]), 2000))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xxx"])


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

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-25 * Du["x"]**2),
                      (Cuboid([0, -0.5], [0, 1]), 100)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u_x"] - 0.1 * Du["u_xx"],
                      (Cuboid([0, -0.5], [1, 1]), 1000))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


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
        self.conditions = [
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

        self.specification = Specification(["u"], ["x", "y"], ["u_xx", "u_yy", "u_xy"])


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

        stack = lambda *tensors: tf.concat(tensors, axis=1)

        initPos = lambda t: tf.concat([0 * t + 1, 0 * t + 0], axis=1)
        initVel = lambda t: tf.concat([0 * t + 0, 0 * t + 0], axis=1)
        ode = lambda u, t, alpha: -alpha * u - tf.concat([0 * t + 0, 0 * t + 1.5], axis=1)

        self.conditions = [
            Condition("initialPos",
                      lambda Du: stack(Du["x"], Du["y"]) - initPos(Du["t"]),
                      (Cuboid([0], [0]), 128)),
            Condition("initialVel",
                      lambda Du: stack(Du["x_t"], Du["y_t"]) - initVel(Du["t"]),
                      (Cuboid([0], [0]), 128)),
            Condition("inner",
                      lambda Du: stack(Du["x_tt"], Du["y_tt"]) - ode(stack(Du["x"], Du["y"]), Du["t"], Du["lagrange"]),
                      (Cuboid([0], [10]), 128)),
            Condition("constraint",
                      lambda Du: tf.norm(stack(Du["x"], Du["y"]), axis=1)**2 - 1.,
                      (Cuboid([0], [10]), 128))
        ]

        self.specification = Specification(["x", "y", "lagrange"], ["t"], ["x_tt", "y_tt"])


if __name__ == "__main__":
    from PDESolver import *

    class OldPendulum(BoundaryValueProblem):
        """
        Class defining the differential algebraic equation for a pendulum.

        t ⟼ (x_1, x_2, lambda)
        """

        def __init__(self):
            """
            Constructor for a pendulum.
            """

            super().__init__()

            initPos = lambda t: tf.concat([0 * t + 1, 0 * t + 0], axis=1)
            initVel = lambda t: tf.concat([0 * t + 0, 0 * t + 0], axis=1)
            ode = lambda u, t, alpha: -alpha * u - tf.concat([0 * t + 0, 0 * t + 1.5], axis=1)

            self.conditions = [
                Condition("initialPos",
                        lambda Du: Du["u"] - initPos(Du["t"]),
                        (Cuboid([0], [0]), 5)),
                Condition("initialVel",
                        lambda Du: Du["u_t"] - initVel(Du["t"]),
                        (Cuboid([0], [0]), 5)),
                Condition("inner",
                        lambda Du: Du["u_tt"] - ode(Du["u"], Du["t"], Du["alpha"]),
                        (Cuboid([0], [10]), 5)),
                Condition("constraint",
                        lambda Du: tf.norm(Du["u"], axis=1)**2 - 1.,
                        (Cuboid([0], [10]), 5))
            ]

            self.specification = Specification(["x", "y", "lagrange"], ["t"], ["x_tt", "y_tt"])

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

    stack = lambda *tensors: tf.concat(tensors, axis=1)
    bvp_new = Pendulum()
    bvp_old = OldPendulum()

    samples = bvp_old.conditions[3].sample_points()

    solver_new = Solver(bvp_new)
    solver_old = Solver(bvp_old)

    Du_new = solver_new.compute_differentials(samples)
    Du_old = bvp_old.calculate_differentials(solver_old.model, samples)

    innerres_new = bvp_new.conditions[2].residue_fn
    innerres_old = bvp_old.conditions[2].residue_fn

    residue_new = innerres_new(Du_new)
    residue_old = innerres_old(Du_old)

    print("Output difference (u): ", tf.abs(Du_old["u"] - stack(Du_new["x"], Du_new["y"])), "\n")
    print("Output difference (u_t): ", tf.abs(Du_old["u_t"] - stack(Du_new["x_t"], Du_new["y_t"])), "\n")
    print("Output difference (u_tt): ", tf.abs(Du_old["u_tt"] - stack(Du_new["x_tt"], Du_new["y_tt"])), "\n")
    print("Lagrange difference: ", tf.abs(Du_old["alpha"] - Du_new["lagrange"]), "\n")
    print("Residue difference: ", tf.abs(residue_old - residue_new), "\n")
