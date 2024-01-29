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
        dataset_size: int = 1000,
        dims: int = 100,
        dim_range: tuple[int, int] = (-5, 5)
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

        n = math.ceil(math.log(dataset_size, 26))
        data_file: Path = output_dir / "rand_dataset.json"
        data_file.unlink(missing_ok=True)
        dataset: list[dict[str, Any]] = []

        for i, identifier in enumerate(cls._iter_identifiers(n), 1):
            embedding: tuple[float, ...] = cls._generate_random_embedding(
                dims, dim_range
            )
            dataset.append({"raw": identifier, "embedding": embedding})

            if i >= dataset_size:
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
        cls, n: int, dim_range: tuple[int, int]
    ) -> tuple[float, ...]:
        """Generates a random embedding using the normal distribution to promote
        clusters.

        Args:
            n: The vector's dimension.
            dim_range: The range of each axe of the vector.

        Returns:
            tuple[float]: The generated vector.
        """

        embedding: list[float] = []

        for _ in range(n):
            embedding.append(
                np.random.normal(loc=random.choice(dim_range), scale=2, size=1)[0]
            )

        return tuple(embedding)
