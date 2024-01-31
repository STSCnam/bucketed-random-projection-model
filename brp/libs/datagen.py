import itertools
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator

import numpy as np


class DatasetGenerator:
    """The random dataset generator class."""

    @classmethod
    def generate(
        cls,
        output_dir: Path,
        *,
        n: int = 100,
        a: float = -5.0,
        b: float = 5.0,
        size: int = 1000,
        num_clusters: int = 5
    ) -> None:
        """Generate a random dataset according to the given specificities.
        Each embedding will be distributed basing of the Gaussian distribution to create
        clusters.

        Args:
            output_dir: The dataset output dir where the JSON file will be saved.
            dataset_size (Optional): The size of the dataset. Default to 1000.
            dims (Optional): The working dimension of the data. Default to 100.
            dim_range (Optional): The range of each axes to distribute the data embeddings.
        """

        n = math.ceil(math.log(size, 26))
        data_file: Path = output_dir / "rand_dataset.json"
        data_file.unlink(missing_ok=True)
        dataset: list[dict[str, Any]] = []
        points_per_cluster: int = int(size / num_clusters)
        sigma: float = (abs(a) + abs(b)) / 10

        for i, identifier in enumerate(cls._iter_identifiers(n)):
            if i % points_per_cluster == 0:
                axePositions: list[float] = list(np.random.uniform(a, b, size=n))

            embedding: tuple[float, ...] = cls._generate_random_embedding(
                n, sigma, axePositions
            )
            dataset.append({"raw": identifier, "embedding": embedding})

            if i >= size - 1:
                break

        data_file.write_text(json.dumps(dataset))

    @classmethod
    def _iter_identifiers(cls, n: int) -> Iterator[str]:
        """Create identifiers based on the capitalized alphabetial letters according.
        The length of the itendifier string is given by the n parameter.

        Args:
            n: The size of the string.

        Returns:
            Iterator[str]: The identifier's combination iterator.
        """

        letters: list[str] = [chr(o) for o in range(65, 91)]

        for combination in itertools.product(letters, repeat=n):
            yield "".join(combination)

    @classmethod
    def _generate_random_embedding(
        cls, n: int, sigma: float, axePositions: list[float]
    ) -> tuple[float, ...]:
        """Generates a random embedding using the normal distribution to promote
        clusters.

        Args:
            n: The vector's dimension.
            sigma: The standard deviation from mu for the normal distribution standard.
            axePositions: The random axe values from space to distribute each vector's
                axe.

        Returns:
            tuple[float]: The generated vector.
        """

        embedding: list[float] = []

        for i in range(n):
            embedding.append(
                np.random.normal(loc=axePositions[i], scale=sigma, size=1)[0]
            )

        return tuple(embedding)
