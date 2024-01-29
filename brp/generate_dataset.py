import sys
from pathlib import Path
from brp.libs.datagen import DatasetGenerator


class Main:
    @classmethod
    def run(cls, ds_size: int, n: int) -> None:
        output_dir: Path = Path("datasets")
        print("Generating dataset...")
        DatasetGenerator.generate(
            output_dir, dataset_size=ds_size, dims=n, dim_range=(-10, 20)
        )


if __name__ == "__main__":
    Main.run(*map(int, sys.argv[1:]))
