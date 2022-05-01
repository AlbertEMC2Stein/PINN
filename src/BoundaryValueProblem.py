from Sampling import *


class Condition:
    def __init__(self, name, residue_fn, region_samples_pair, sampler=Equidistant(), weight=1):
        self.name = name
        self.residue_fn = residue_fn
        self.sample_points = lambda: region_samples_pair[0].pick(region_samples_pair[1], sampler)
        self._region = region_samples_pair[0]
        self.weight = weight

    def __call__(self, model, bvp):
        Du = bvp.calculate_differentials(model, self.sample_points())
        return self.residue_fn(Du)

    def get_region_bounds(self):
        return self._region.get_bounds()


class BoundaryValueProblem:
    conditions = []

    @staticmethod
    def calculate_differentials(model, freeVariables):
        ...


class WaveEquation1D(BoundaryValueProblem):
    conditions = [
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


class BurgersEquation(BoundaryValueProblem):
    conditions = [
        Condition("initial",
                  lambda Du: Du["u"] + tf.sin(np.pi * Du["x"]),
                  (Cuboid([0, -1], [0, 1]), 100)),
        Condition("boundary",
                  lambda Du: Du["u"],
                  (Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1])), 100)),
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
    conditions = [
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
    conditions = [
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
    conditions = [
        Condition("initial",
                  lambda Du: Du["u"] - Du["x"] ** 2 * tf.cos(np.pi * Du["x"]) - 1,
                  (Cuboid([0, -1], [0, 1]), 100),
                  weight=10),
        Condition("boundary",
                  lambda t, x, Du: Du["u_x"],
                  (Union(Cuboid([0, -1], [1, -1]), Cuboid([0, 1], [1, 1])), 200)),
        Condition("inner",
                  lambda t, x, Du: Du["u_t"] - 0.33 * Du["u_xx"],
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


class MinimalSurfaceEquation(BoundaryValueProblem):
    conditions = [
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
