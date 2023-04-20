from PDESolver.Sampling import *
import tensorflow as tf


class Specification:
    def __init__(self, components, variables, differentials, stacked_components={}):
        """
        Class for defining the differential specification of a boundary value problem.

        Parameters
        -----------
        components: list
            List of names for the output components
        variables: list
            List of names for the input variables
            Remark: As of right now only single character variables are supported
        differentials: list
            List of differentials needed to check the constraints
        stacked_components: dict
            Dictionary of stacked components to make constraint formulation easier

        Example
        -----------
        Consider the EOM of a simple pendulum using cartesian coordinates:

        >>> Specification(["x", "y", "lagrange"], # Components of the output
        ...               ["t"], # Input variable
        ...               ["x_tt", "y_tt"], # Differentials needed to check the constraints
        ...               {"u": ["x", "y"], "u_t": ["x_t", "y_t"], "u_tt": ["x_tt", "y_tt"]}) # Stacked components 
        >>> Condition("algebraic constraint",
        ...           lambda Du: tf.reshape(tf.norm(Du["u"], axis=1), (-1, 1))**2 - 1.,
        ...           (Cuboid([0], [1]), 128))
        """

        self.components = components
        self.variables = variables
        self.differentials = differentials
        self.stacked_components = stacked_components

    def as_dictionary(self):
        """
        Returns the specification as a dictionary.

        Returns
        -----------
        dict: {"components": [str], "variables": [str], "differentials": [str], "stacked_components": {str: [str]}
        """

        return {"components": self.components, "variables": self.variables, "differentials": self.differentials, "stacked_components": self.stacked_components}


class Condition:
    def __init__(self, name, residue_fn, region_samples_pair, sampler=Random()):
        """
        Class for defining a boundary value problem condition/constraint.

        Parameters
        -----------
        name: str
            Name of the condition
        residue_fn: function
            Function that takes in a dictionary of differentials and returns the residue
        region_samples_pair: tuple
            Tuple of (Region, int) where the Region is the region where the condition is applied and the int is the number of samples to take from the region
        sampler: Sampler
            Sampler to use for sampling from the region

        Example
        -----------
        >>> Condition("Laplace", # Name of the condition
        ...           lambda Du: Du["u_xx"] + Du["u_yy"], # Residue function
        ...           (Cuboid([0, 0], [1, 1]), 128), # Unit square together with number of samples
        ...           sampler=Random()) # Random sampler

        >>> Condition("HeatEquation", # Name of the condition
        ...           lambda Du: Du["u_t"] - Du["u_xx"], # Residue function
        ...           (Cuboid([0, 0], [1, 1]), 128), # Unit square together with number of samples
        ...           sampler=Random()) # Random sampler
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
    def __init__(self, minibatch_size=128):
        """
        Class for defining a boundary value problem.

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
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

        Returns
        -----------
        dict: Specification of the boundary value problem
        """

        return self.specification.as_dictionary()


class Laplace(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the Laplace equation ( \\(\\Omega = [0, 1]^2 \\) ).
        \\begin{alignat}{2}
            \\Delta u &= 0 \\quad &&\\text{in } \\Omega \\newline
            u(x, y) &= 0 \\quad &&\\forall (x, y) \\in \\Gamma_S \\cup \\Gamma_W \\cup \\Gamma_E \\newline
            u(x, y) &= 2x(1 - x) \\quad &&\\forall (x, y) \\in \\Gamma_N
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)
 
        self.conditions = [
            Condition("zero_boundary",
                      lambda Du: Du["u"],
                      (Cuboid([0, 0], [0, 1]) & Cuboid([0, 0], [1, 0]) & Cuboid([1, 0], [1, 1]), minibatch_size)),

            Condition("f_boundary",
                      # tf.where(tf.abs(Du["x"] - 0.5) < 0.25, 1., 0.),
                      lambda Du: Du["u"] - 2 * Du["x"] * (1 - Du["x"]),
                      (Cuboid([0, 1], [1, 1]), minibatch_size)),

            Condition("inner",
                      lambda Du: Du["u_xx"] + Du["u_yy"],
                      (Cuboid([0, 0], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["x", "y"], ["u_xx", "u_yy"])


class WaveEquation1D(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the wave equation in one spatial dimension ( \\(\\Omega = [0, 1] \\times [-1, 1]\\) ).
        \\begin{alignat}{2}
            u_{tt} &= u_{xx} \\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= \\exp(-25x^2) \\quad &&\\forall x \\in [-1, 1] \\newline
            u_{t}(0, x) &= 0 \\quad &&\\forall x \\in [-1, 1]
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)
        
        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-25 * Du["x"]**2), #1 / (200*(Du["x"] + 0.5) ** 2 + 1) - 1 / (200*(Du["x"] - 0.5)**2 + 2),
                      (Cuboid([0, -1], [0, 1]), minibatch_size)),
            Condition("boundary1",
                      lambda Du: Du["u_t"],
                      (Cuboid([0, -1], [0, 1]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_tt", "u_xx"])


class WaveEquation2D(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the wave equation in two spatial dimensions ( \\(\\Omega = [0, 1] \\times [-2, 2]^2\\) ).
        \\begin{alignat}{2}
            u_{tt} &= u_{xx} \\quad &&\\text{in } \\Omega \\newline
            u(0, x, y) &= \\exp(-x^2 - y^2)^4 \\quad &&\\forall (x, y) \\in [-2, 2]^2 \\newline
            u_{t}(0, x, y) &= 0 \\quad &&\\forall (x, y) \\in [-2, 2]^2
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial_u",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 4,
                      (Cuboid([0, -2, -2], [0, 2, 2]), minibatch_size)),
            Condition("initial_u",
                      lambda Du: Du["u_t"],
                      (Cuboid([0, -2, -2], [0, 2, 2]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, -2, -2], [2, 2, 2]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x", "y"], ["u_tt", "u_xx", "u_yy"])


class HeatEquation1D(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the heat equation in one spatial dimension ( \\(\\Omega = [0, 1]^2\\) ).

        \\begin{alignat}{2}
            u_{t} &= \\frac{1}{8} u_{xx} \\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= 8x^2 (1 - x)^2 \\quad &&\\forall x \\in [0, 1] \\newline
            u_{x}(t, x) &= 0 \\quad &&\\forall (t, x) \\in [0, 1] \\times \\lbrace 0, 1 \\rbrace
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - 8*Du["x"]**2 * (1 - Du["x"])**2,
                      (Cuboid([0, 0], [0, 1]), minibatch_size)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (Cuboid([0, 0], [1, 0]) & Cuboid([0, 1], [1, 1]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.125 * Du["u_xx"],
                      (Cuboid([0, 0], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


class HeatEquation2D(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the heat equation in two spatial dimensions ( \\(\\Omega = [0, 1] \\times [-2, 2]^2\\) ).

        \\begin{alignat}{2}
            u_{t} &= u_{xx} + u_{yy} \\quad &&\\text{in } \\Omega \\newline
            u(0, x, y) &= \\exp(-x^2 - y^2)^2 \\quad &&\\forall (x, y) \\in [-2, 2]^2
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-Du["x"] ** 2 - Du["y"] ** 2) ** 2,
                      (Cuboid([0, -2, -2], [0, 2, 2]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_t"] - Du["u_xx"] - Du["u_yy"],
                      (Cuboid([0, -2, -2], [2, 2, 2]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x", "y"], ["u_t", "u_xx", "u_yy"])


class BurgersEquation(BoundaryValueProblem):
    def __init__(self, viscosity=0.01, minibatch_size=128):
        """
        Class defining the Burgers equation in one spatial dimension ( \\(\\Omega = [0, 1] \\times [-1, 1]\\) ).

        \\begin{alignat}{2}
            u_{t} + u u_{x} &= \\frac{\\nu}{\\pi} u_{xx} \\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= -\\sin(\\pi x) \\quad &&\\forall x \\in [-1, 1] \\newline
            u(t, x) &= 0 \\quad &&\\forall (t, x) \\in [0, 1] \\times \\lbrace -1, 1 \\rbrace
        \\end{alignat}

        Parameters
        -----------
        viscosity: float
            Viscosity parameter
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)
        self.viscosity = viscosity

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] + tf.sin(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), minibatch_size)),
            Condition("boundary",
                      lambda Du: Du["u"],
                      (Cuboid([0, -1], [1, -1]) & Cuboid([0, 1], [1, 1]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u"] * Du["u_x"] - self.viscosity / np.pi * Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), minibatch_size))
        ]
        
        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


class VanDerPolEquation(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the Van der Pol equation with varying \\(\\mu\\) ( \\(\\Omega = [0, 4] \\times [0, 10]\\) ).

        \\begin{alignat}{2}
            u_{xx} - t(1 - u^2)u_{x} + u &= 0 \\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= \\cos(x) \\quad &&\\forall x \\in [0, 10] \\newline
            u(t, 0) &= 1 \\quad &&\\forall t \\in [0, 4] \\newline
            u_{x}(t, 0) &= 0 \\quad &&\\forall t \\in [0, 4]
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - 1,
                      (Cuboid([0, 0], [4, 0]), minibatch_size)),
            Condition("boundary",
                      lambda Du: Du["u_x"],
                      (Cuboid([0, 0], [4, 0]), minibatch_size)),
            Condition("helper",
                      lambda Du: Du["u"] - tf.cos(Du["x"]),
                      (Cuboid([0, 0], [0, 10]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_xx"] - Du["t"] * (1 - Du["u"] ** 2) * Du["u_x"] + Du["u"],
                      (Cuboid([0, 0], [4, 10]), minibatch_size))
        ]
        
        self.specification = Specification(["u"], ["t", "x"], ["u_xx"])


class AllenCahnEquation(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the Allen-Cahn equation in one spatial dimension ( \\(\\Omega = [0, 1] \\times [-1, 1]\\) ).

        \\begin{alignat}{2}
            u_{t} &= 0.0001u_{xx} - 5(u^3 - u) \\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= -x^2\\cos(\\pi x) \\quad &&\\forall x \\in [-1, 1] \\newline
            u(t, x) &= -1 \\quad &&\\forall (t, x) \\in [0, 1] \\times \\lbrace -1, 1 \\rbrace \\newline
            u_{x}(t, x) &= 0 \\quad &&\\forall (t, x) \\in [0, 1] \\times \\lbrace -1, 1 \\rbrace \\newline
            u(t, 0) &= 0 \\quad &&\\forall t \\in [0, 1]
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """


        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - Du["x"] ** 2 * tf.cos(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), minibatch_size)),
            Condition("boundary1",
                      lambda Du: Du["u"] + 1,
                      (Cuboid([0, -1], [1, -1]) & Cuboid([0, 1], [1, 1]), minibatch_size)),
            Condition("boundary2",
                      lambda Du: Du["u_x"],
                      (Cuboid([0, -1], [1, -1]) & Cuboid([0, 1], [1, 1]), minibatch_size)),
            Condition("center",
                      lambda Du: Du["u"],
                      (Cuboid([0, 0], [1, 0]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_t"] - 0.0001 * Du["u_xx"] + 5 * (Du["u"] ** 3 - Du["u"]),
                      (Cuboid([0, -1], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


class KortewegDeVriesEquation(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the Korteweg-de Vries equation in one spatial dimension ( \\(\\Omega = [0, 1] \\times [-1, 1]\\) ).

        \\begin{alignat}{2}
            u_{t} - 6u u_{x} - u_{xxx} &= 0\\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= \\cos(\\pi x) \\quad &&\\forall x \\in [-1, 1]
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.cos(np.pi * Du["x"]),
                      (Cuboid([0, -1], [0, 1]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_t"] + 6 * Du["u"] * Du["u_x"] + Du["u_xxx"],
                      (Cuboid([0, -1], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xxx"])


class ConvectionDiffusionEquation(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the convection-diffusion equation in one spatial dimension ( \\(\\Omega = [0, 1] \\times [-0.5, 1]\\) ).

        \\begin{alignat}{2}
            u_{t} &= 0.1u_{xx} - u_{x} \\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= \\exp(-25x^2) \\quad &&\\forall x \\in [-0.5, 1]
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("initial",
                      lambda Du: Du["u"] - tf.exp(-25 * Du["x"]**2),
                      (Cuboid([0, -0.5], [0, 1]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_t"] + Du["u_x"] - 0.1 * Du["u_xx"],
                      (Cuboid([0, -0.5], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])


class MinimalSurfaceEquation(BoundaryValueProblem):
    def __init__(self, minibatch_size=128):
        """
        Class defining the minimal surface equation in two spatial dimensions ( \\(\\Omega = [0, 1] \\times [-1, 1]\\) ).

        \\begin{alignat}{2}
            (1 + u_{x}^2)u_{yy} - 2u_{y}u_{x}u_{xy} + (1 + u_{y}^2)u_{xx} &= 0\\quad &&\\text{in } \\Omega \\newline
            u(0, x) &= x \\quad &&\\forall x \\in [-1, 1] \\newline
            u(1, x) &= -x \\quad &&\\forall x \\in [-1, 1] \\newline
        \\end{alignat}

        Parameters
        -----------
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        self.conditions = [
            Condition("boundary1",
                      lambda Du: Du["u"] - Du["x"],
                      (Cuboid([0, -1], [0, 1]), minibatch_size)),
            Condition("boundary2",
                      lambda Du: Du["u"] + Du["x"],
                      (Cuboid([1, -1], [1, 1]), minibatch_size)),
            Condition("inner",
                      lambda Du: (1 + Du["u_x"] ** 2) * Du["u_yy"] - 2 * Du["u_y"] * Du["u_x"] * Du["u_xy"] + (
                              1 + Du["u_y"] ** 2) * Du["u_xx"],
                      (Cuboid([0, -1], [1, 1]), minibatch_size))
        ]

        self.specification = Specification(["u"], ["x", "y"], ["u_xx", "u_yy", "u_xy"])


class Pendulum(BoundaryValueProblem):
    def __init__(self, init_pos=[1, 0], init_vel=[0, 0], gravity=1.5, t_start=0, t_end=2, minibatch_size=128):
        """
        Class defining the EOM for the simple pendulum ( \\(\\Omega = [T_0, T_1]\\) ).

        \\begin{alignat}{2}
            u_{tt} &= -\\lambda u - (0, g)^\\top \\quad &&\\text{in } \\Omega \\newline
            \\Vert u \\Vert_2^2 &= 1 \\quad &&\\text{in } \\Omega \\newline
            u(0) &= p_0 \\newline
            u_{t}(0) &= v_0 \\newline
        \\end{alignat}

        Parameters
        -----------
        init_pos: list
            Initial position of the pendulum
        init_vel: list
            Initial velocity of the pendulum
        gravity: float
            Gravity constant    
        t_start: float
            Start time
        t_end: float
            End time
        minibatch_size: int
            Size of the minibatch to use for training
        """

        super().__init__(minibatch_size)

        initPos = lambda t: tf.concat([0*t + init_pos[0], 0*t + init_pos[1]], axis=1)
        initVel = lambda t: tf.concat([0*t + init_vel[0], 0*t + init_vel[1]], axis=1)
        ode = lambda u, t, alpha: -alpha * u - tf.concat([0*t + 0, 0*t + gravity], axis=1)

        self.conditions = [
            Condition("initialPos",
                      lambda Du: Du["u"] - initPos(Du["t"]),
                      (Cuboid([t_start], [t_start]), minibatch_size)),
            Condition("initialVel",
                      lambda Du: Du["u_t"] - initVel(Du["t"]),
                      (Cuboid([t_start], [t_start]), minibatch_size)),
            Condition("inner",
                      lambda Du: Du["u_tt"] - ode(Du["u"], Du["t"], Du["lagrange"]),
                      (Cuboid([t_start], [t_end]), minibatch_size)),
            Condition("constraint",
                      lambda Du: tf.reshape(tf.norm(Du["u"], axis=1), (-1, 1))**2 - 1.,
                      (Cuboid([t_start], [t_end]), minibatch_size))
        ]

        self.specification = Specification(["x", "y", "lagrange"], ["t"], ["x_tt", "y_tt"], 
                                           {"u": ["x", "y"], "u_t": ["x_t", "y_t"], "u_tt": ["x_tt", "y_tt"]})
