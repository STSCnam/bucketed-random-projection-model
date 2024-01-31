import sys
from pathlib import Path
from brp.libs.datagen import DatasetGenerator


class Main:
    @classmethod
    def run(cls, size: int, n: int) -> None:
        output_dir: Path = Path("datasets")
        print("Generating dataset...")
        DatasetGenerator.generate(
            output_dir, n=n, a=-50.0, b=50.0, size=size, num_clusters=3
        )


if __name__ == "__main__":
    Main.run(*map(int, sys.argv[1:]))
