from collections import defaultdict
from functools import cached_property
import json
from pathlib import Path
import numpy as np
from brp.libs.indexer import Bucket, Data, Hyperplane, Index
from brp.libs.types import Vector


class _BRPModel:
    """The BRP model implementation.

    Attributes:
        num_hyperplanes: The number of hyperplanes that'll be used to build the model.
        bucket_size: The length of the hyperplane's bucket.
        index: The index instance.
        hyperplanes: The random hyperplanes.
    """

    def __init__(self, index: Index, num_hyperplanes: int, bucket_size: float) -> None:
        """The model initializer.

        Args:
            index: The index instance.
            num_hyperplanes: The number of hyperplanes that'll be used to build the model.
            bucket_size: The length of the hyperplane's bucket.
        """

        self.num_hyperplanes: int = num_hyperplanes
        self.bucket_size: float = bucket_size
        self._index: Index = index
        self.hyperplanes: list[Hyperplane] = []

    def generate_buckets(self) -> None:
        """Generate buckets for each data."""

        self._index.clean(Bucket, Hyperplane)

        for data in self._index.fetch_all_data():
            if not self.hyperplanes:
                for hyperplane in self._gen_random_hyperplanes(len(data.embedding)):
                    self.hyperplanes.append(
                        self._index.create(Hyperplane(vector=tuple(hyperplane)))
                    )

            Path("datasets/hyperplanes.json").write_text(
                json.dumps([h.vector for h in self.hyperplanes]), encoding="utf-8"
            )
            hash_set: Vector = self._compute_hash_set(data.embedding)
            flattened_hash: int = self._flatten_hashset(hash_set)
            bucket: Bucket = Bucket(hash=flattened_hash)
            bucket = self._index.create(bucket)
            data.bucket = bucket
            self._index.update(data)

    def get_approximate_nearest_neighbors(
        self, query: Vector, k: int = 1
    ) -> list[tuple[float, str]]:
        """Query for the approximated nearest neighbors from the indexed dataset.

        Args:
            query: The data's embedding to query.
            k: The k nearest neighbors to return.

        Returns:
            list[tuple[float, str]]: The k nearest neighbords of the query.
        """

        hash_index: int = self._flatten_hashset(self._compute_hash_set(query))
        bucket: Bucket = self._index.fetch_bucket(hash_index)
        bucket_distances: list[tuple[float, Vector]] = []

        for data in self._index.fetch_bucket_data(bucket):
            v: np.array = np.array(data.embedding)
            qv: np.array = np.array(query)
            bucket_distances.append((np.linalg.norm(qv - v), data.raw))

        return sorted(bucket_distances, key=lambda x: x[0])[:k]

    def load_hyperplanes(self) -> None:
        """Fetch hyperplanes from db and load them into the model instance."""

        self.hyperplanes = list(self._index.fetch_all_hyperplanes())
        self.num_hyperplanes = len(self.hyperplanes)

    def _compute_hash_set(self, v: Vector) -> Vector:
        """Determine the hash set according to the given point.

        Args:
            v: The point to transform.

        Returns:
            Vector: The list of hashes per hyperplane.
        """

        v_hashes: Vector = []

        for hyperplane in self.hyperplanes:
            v_hashes.append(self._compute_hash(v, hyperplane.vector, self.bucket_size))

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

    def load_model(self, index: Index, force_init: bool = False) -> _BRPModel:
        """Build the BRP model according to the given points.

        Args:
            index: The index instance.
            force_init: A boolean that specify if the model should regenerate
                random hyperplanes and buckets. Default to False.

        Returns:
            _BRPModel: The built BRP model.
        """

        model: _BRPModel = _BRPModel(index, self.num_hyperplanes, self.bucket_size)

        if force_init:
            model.generate_buckets()
        else:
            model.load_hyperplanes()

        return model
