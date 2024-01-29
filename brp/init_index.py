import json
import sys
from typing import Any
from pathlib import Path
from brp.libs.indexer import Index
from brp.libs.lsh import BucketedRandomProjection


class Main:
    @classmethod
    def run(cls, num_hyperplanes: int, bucket_size: float) -> None:
        dataset_file: Path = Path("datasets/rand_dataset.json")
        db_file: Path = Path(".databases/.index.sqlite3")
        dataset: list[dict[str, Any]] = json.load(
            dataset_file.open("r", encoding="utf-8")
        )
        index: Index = Index(db_file, force_init=True)
        print(f"Populating index from {dataset_file}...")
        index.populate(dataset)
        brp: BucketedRandomProjection = BucketedRandomProjection(
            num_hyperplanes=num_hyperplanes, bucket_size=bucket_size
        )
        print(
            f"Initializing model with {num_hyperplanes} hyperplanes "
            f"and a bucket size of {bucket_size}..."
        )
        brp.load_model(index, force_init=True)


if __name__ == "__main__":
    Main.run(int(sys.argv[1]), float(sys.argv[2]))
