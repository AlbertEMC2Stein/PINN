import tensorflow as tf
import numpy as np


class Region:
    def __init__(self):
        """
        Constructor for a Region class.
        """

        pass

    def pick(self, n, sampler, ignore_samples):
        """
        Samples n points from the region.

        Parameters
        -----------
        n: int
            Number of points to sample
        sampler: Sampler
            Sampler to use
        ignore_samples: bool
            If True, the samples are not stored the first time pick is called

        Returns
        -----------
        array: Array of shape (n, dimension)
        """
        ...

    def get_bounds(self):
        """
        Returns the bounds of the region.

        Returns
        -----------
        array: Array of shape (2, dimension)
        """
        ...


class Cuboid(Region):
    def __init__(self, corner1, corner2):
        """
        Constructor for a cuboid region which sides are parallel to all axis

        Parameters
        -----------
        corner1: list
            Coordinates of the first corner
        corner2: list
            Coordinates of the second corner
        """

        super().__init__()
        self.corner1 = tf.constant(corner1, dtype=tf.float32)
        self.corner2 = tf.constant(corner2, dtype=tf.float32)
        self.diff_idx = tf.reshape(tf.where(self.corner1 != self.corner2), [-1])
        self.dimension = len(self.diff_idx)
        self.samples = None

    def pick(self, n, sampler, ignore_samples=False):
        if self.samples is not None and sampler.isPersistent:
            tmp = self.samples
        elif self.dimension == 0:
            tmp = tf.ones((n, len(self.corner1))) * self.corner1
        else:
            start = tf.gather(self.corner1, self.diff_idx)
            end = tf.gather(self.corner2, self.diff_idx)

            k = tf.cast(tf.math.ceil(n**(1 / self.dimension)), tf.int64)
            lines = []
            for i in range(self.dimension):
                ts = sampler.pick(k)
                lines += [(1 - ts) * start[i] + ts * end[i]]

            xy_matrix = tf.meshgrid(*lines)
            xy_vector = tf.concat([tf.reshape(variable, [-1, 1]) for variable in xy_matrix], axis=-1)

            n_approx = k**self.dimension
            tmp = tf.ones((n_approx, len(self.corner1))) * self.corner1

            row_idx = tf.range(n_approx, dtype=tf.int64)
            row_idx = tf.repeat(row_idx, self.dimension)

            col_idx = tf.transpose(tf.tile(self.diff_idx[:, tf.newaxis], [1, n_approx]))
            col_idx = tf.reshape(col_idx, [-1])

            indices = tf.transpose(tf.reshape(tf.concat([row_idx, col_idx], 0), [-1, n_approx * self.dimension]))
            tmp = tf.tensor_scatter_nd_update(tmp, indices, tf.reshape(xy_vector, [-1]))

            if self.samples is None and not ignore_samples:
                self.samples = tmp

        return tmp

    def get_bounds(self):
        return self.corner1.numpy(), self.corner2.numpy()


class Union(Region):
    def __init__(self, *regions):
        """
        Constructor for a union of regions.

        Parameters
        -----------
        regions: Region
            List of regions
        """

        super().__init__()
        self.regions = regions

    def pick(self, n, sampler, ignore_samples=False):
        num_regions = len(self.regions)
        ns = [n // num_regions] * num_regions
        ns[-1] += n % num_regions

        return tf.concat([r.pick(ns[i], sampler, ignore_samples) for i, r in enumerate(self.regions)], axis=0)

    def get_bounds(self):
        upper = tf.maximum([np.max(region.get_bounds()) for region in self.regions], axis=0)
        lower = tf.minimum([np.min(region.get_bounds()) for region in self.regions], axis=0)

        return lower, upper


class Sampler:
    def __init__(self):
        """
        Constructor for a Sampler class.

        Attributes
        -----------
        isPersistent: boolean
            Indicating whether the sampler is persistent
        """

        self.isPersistent = None

    def pick(self, n):
        """
        Samples n points from [0; 1] following a pattern determined by the used Sampler

        Parameters
        -----------
        n: int
            Number of points to sample

        Returns
        -----------
        array: Array of shape (n,)
        """
        ...


class Random(Sampler):
    def __init__(self):
        """
        Constructor for a Random Sampler class.

        Attributes
        -----------
        isPersistent: boolean
            Indicating whether the sampler is persistent
        """

        super().__init__()
        self.isPersistent = False

    def pick(self, n):
        """
        Samples n uniformly distributed points from [0; 1].

        Parameters
        -----------
        n: int
            Number of points to sample

        Returns
        -----------
        array: Array of shape (n,)
        """

        return tf.random.uniform(shape=(1,n))


class FirstRandom(Sampler):
    def __init__(self):
        """
        Constructor for a FirstRandom Sampler class.

        Attributes
        -----------
        isPersistent: boolean
            Indicating whether the sampler is persistent
        """

        super().__init__()
        self.isPersistent = True

    def pick(self, n):
        """
        Samples n uniformly distributed points from [0; 1].
        Same as Random but the with isPersistent=True.

        Parameters
        -----------
        n: int
            Number of points to sample

        Returns
        -----------
        array: Array of shape (n,)
        """
        return tf.random.uniform(shape=(1,n))


class Equidistant(Sampler):
    def __init__(self):
        """
        Constructor for a Equidistant Sampler class.

        Attributes
        -----------
        isPersistent: boolean
            Indicating whether the sampler is persistent
        """

        super().__init__()
        self.isPersistent = True

    def pick(self, n):
        """
        Samples n equidistant points from [0; 1].

        Parameters
        -----------
        n: int
            Number of points to sample

        Returns
        -----------
        array: Array of shape (n,)
        """
        return tf.linspace(0, 1, n)


class EquidistantRandom(Sampler):
    def __init__(self, n):
        """
        Constructor for a EquidistantRandom Sampler class.

        Attributes
        -----------
        isPersistent: boolean
            Indicating whether the sampler is persistent
        """

        super().__init__()
        self.isPersistent = False
        self.n = n
        self.original = tf.linspace(0, 1, int(n))

    def pick(self, n):
        """
        Samples n equidistant points from [0; 1] with a random offset.

        Parameters
        -----------
        n: int
            Number of points to sample

        Returns
        -----------
        array: Array of shape (n,)
        """

        idx = tf.random.uniform(shape=(n,), minval=0, maxval=self.n - 1, dtype=tf.int64)
        return tf.gather(self.original, idx)