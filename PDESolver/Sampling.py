import tensorflow as tf
import numpy as np


class Region:
    def __init__(self):
        pass

    def pick(self, n, sampler):
        ...

    def get_bounds(self):
        ...


class Cuboid(Region):
    def __init__(self, corner1, corner2):
        super().__init__()
        self.corner1 = np.array(corner1)
        self.corner2 = np.array(corner2)
        self.diff_idx = np.where(self.corner1 != self.corner2)[0]
        self.dimension = self.diff_idx.size
        self.samples = None

        if self.dimension == 0:
            self.diff_idx = np.array([0])
            self.dimension = 1

    def pick(self, n, sampler):
        if self.samples is not None and sampler.isPersistent:
            tmp = self.samples
        else:
            start = self.corner1[self.diff_idx]
            end = self.corner2[self.diff_idx]

            k = int(np.ceil(n**(1 / self.dimension)))
            lines = []
            for i in range(self.dimension):
                ts = sampler.pick(k)
                lines += [(1 - ts) * start[i] + ts * end[i]]

            x = np.meshgrid(*lines)
            y = np.stack(list(map(np.ravel, x))).T

            tmp = np.ones((len(y), len(self.corner1))) * self.corner1
            tmp[:, self.diff_idx] = y

            if self.samples is None:
                self.samples = tmp

        return tf.constant(tmp, "float32")

    def get_bounds(self):
        return self.corner1, self.corner2


class Union(Region):
    def __init__(self, *regions):
        super().__init__()
        self.regions = regions

    def pick(self, n, sampler):
        num_regions = len(self.regions)
        ns = [n // num_regions] * num_regions
        ns[-1] += n % num_regions

        return tf.concat([r.pick(ns[i], sampler) for i, r in enumerate(self.regions)], axis=0)

    def get_bounds(self):
        upper = tf.maximum([np.max(region.get_bounds()) for region in self.regions], axis=0)
        lower = tf.minimum([np.min(region.get_bounds()) for region in self.regions], axis=0)

        return lower, upper


class Sampler:
    def __init__(self):
        self.isPersistent = None

    def pick(self, n):
        ...


class Random(Sampler):
    def __init__(self):
        super().__init__()
        self.isPersistent = False

    def pick(self, n):
        return np.random.uniform(size=(1,n))


class FirstRandom(Sampler):
    def __init__(self):
        super().__init__()
        self.isPersistent = True

    def pick(self, n):
        return np.random.uniform(size=(1,n))


class Equidistant(Sampler):
    def __init__(self):
        super().__init__()
        self.isPersistent = True

    def pick(self, n):
        return np.linspace(0, 1, n)
