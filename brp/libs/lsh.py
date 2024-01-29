from collections import defaultdict
from functools import cached_property
import numpy as np
from brp.libs.indexer import Bucket, Data, Index
from brp.libs.types import Vector


class _BRPModel:
    """The BRP model implementation.

    Attributes:
        num_hyperplanes: The number of hyperplanes that'll be used to build the model.
        bucket_size: The length of the hyperplane's bucket.
        points: The dataset to fit to the model.
        hyperplanes: The randomized hyperplanes.
    """

    def __init__(self, index: Index, num_hyperplanes: int, bucket_size: float) -> None:
        """The model initializer.

        Args:
            points: The dataset to fit to the model.
            num_hyperplanes: The number of hyperplanes that'll be used to build the model.
            bucket_size: The length of the hyperplane's bucket.
        """

        self.num_hyperplanes: int = num_hyperplanes
        self.bucket_size: float = bucket_size
        self._index: Index = index
        self.hyperplanes: list[np.ndarray] | None = None

    def generate_buckets(self) -> None:
        for data in self._index.fetch_all_data():
            if self.hyperplanes is None:
                print("Hello")
                self.hyperplanes = self._gen_random_hyperplanes(len(data.embedding))

            hash_set: Vector = self._compute_hash_set(data.embedding)
            flattened_hash: int = self._flatten_hashset(hash_set)
            bucket: Bucket = Bucket(hash=flattened_hash)
            bucket = self._index.create(bucket)
            data.bucket = bucket
            self._index.update(data)

    def get_approximate_nearest_neighbors(
        self, query: Vector, k: int = 1
    ) -> list[tuple[float, Vector]]:
        """Query for the approximated nearest neighbors from the indexed dataset.

        Args:
            query: The query to compare.
            k: The k neighbors to return.

        Returns:
            list[tuple[float, Point]]: The k nearest neighbords of the query.
        """

        hash_index: int = self._flatten_hashset(self._compute_hash_set(query))
        bucket: Bucket = Bucket(hash=hash_index)
        bucket_distances: list[tuple[float, Vector]] = []

        for data in self._index.fetch_bucket_data(bucket):
            v: np.array = np.array(data.embedding)
            qv: np.array = np.array(query)
            bucket_distances.append((np.linalg.norm(qv - v), data.raw))

        return sorted(bucket_distances, key=lambda x: x[0])[:k]

    # @cached_property
    # def hashes(self) -> dict[int, list[Vector]]:
    #     """Compute the hash set for each point of the dataset.

    #     Returns:
    #         dict[int, list[Point]]: The hash map which is a pair of
    #             (<flattened_hashset>, <points>).
    #     """

    #     hashes: dict[Vector, Vector] = defaultdict(list)

    #     for v in self.dataset:
    #         hashes[self._flatten_hashset(self._compute_hash_set(v))].append(v)

    #     return hashes

    def _compute_hash_set(self, v: Vector) -> Vector:
        """Determine the hash set according to the given point.

        Args:
            v: The point to transform.

        Returns:
            Vector: The list of hashes per hyperplane.
        """

        v_hashes: Vector = []

        for w in self.hyperplanes:
            v_hashes.append(self._compute_hash(v, w, self.bucket_size))

        return v_hashes

    def _gen_random_hyperplanes(self, n: int) -> list[np.ndarray]:
        """Generate random hyperplanes.

        Args:
            n: The number of dimensions.

        Returns:
            list[np.ndarray]: The set of generated hyperplanes.
        """

        hyperplanes: list[np.ndarray] = []

        for _ in range(self.num_hyperplanes):
            w: np.ndarray = np.random.randn(n)
            w /= np.linalg.norm(w)
            hyperplanes.append(w)

        return hyperplanes

    def _flatten_hashset(self, u: Vector) -> int:
        """Reduce the hash set vector into a single dimension repsented by an index.

        Args:
            u: The hash set to reduce.

        Returns:
            int: The computed index of u.
        """

        index = 0

        for i, n in enumerate(range(len(u) - 1, 0, -1)):
            index += (u[i] * 2) ** n

        return index + u[-1]

    def _compute_hash(self, v: Vector, w: np.ndarray, r: float) -> int:
        """Compute the hash of the point v according to the hyperplane w.

        Args:
            v: The point to compute.
            w: The normal hyperplane vector.
            r: The bucket length.

        Returns:
            int: The computed point's hash.
        """

        return int(np.floor(np.dot(v, w) / r))


class BucketedRandomProjection:
    """The BRP model initializer class.

    Attributes:
        num_hyperplanes: The number of hyperplanes that'll be used to build the model.
            Default to 1.
        bucket_size: The length of the hyperplane's bucket. Default to 1.0.
    """

    def __init__(self, num_hyperplanes: int = 1, bucket_size: float = 1.0) -> None:
        """Initializes the BRP parameters.

        Args:
            num_hyperplanes: The number of hyperplanes that'll be used to build the model.
                Default to 1.
            bucket_size: The length of the hyperplane's bucket. Default to 1.0.
        """

        self.num_hyperplanes: int = num_hyperplanes
        self.bucket_size: float = bucket_size

    def build_model(self, index: Index, force_init: bool = False) -> _BRPModel:
        """Build the BRP model according to the given points.

        Args:
            points: The dataset to fit to the model.

        Returns:
            _BRPModel: The built BRP model.
        """

        model: _BRPModel = _BRPModel(index, self.num_hyperplanes, self.bucket_size)

        if force_init:
            model.generate_buckets()

        return model
