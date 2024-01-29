import itertools
import json
from pathlib import Path
from random import randint, random
from typing import Any, Iterator


class DatasetGenerator:
    @classmethod
    def generate(cls, output_dir: Path, n: int) -> None:
        data_file: Path = output_dir / "rand_dataset.json"
        dataset: list[dict[str, Any]] = []

        for identifier in cls._iter_identifiers(n):
            embedding: tuple[float, ...] = cls._generate_random_embedding(100)
            dataset.append({"raw": identifier, "embedding": embedding})

        json.dump(dataset, data_file.open("w", encoding="utf-8"))

    @classmethod
    def _iter_identifiers(cls, n: int) -> Iterator[str]:
        letters: list[str] = [chr(o) for o in range(65, 91)]

        for combination in itertools.product(letters, repeat=n):
            yield "".join(combination)

    @classmethod
    def _generate_random_embedding(cls, n: int) -> tuple[float, ...]:
        embedding: list[float] = []

        for _ in range(n):
            embedding.append(randint(-5, 5) + random())

        return tuple(embedding)
