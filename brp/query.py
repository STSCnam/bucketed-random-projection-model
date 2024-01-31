import sys
from pprint import pp
from pathlib import Path
from brp.libs.indexer import Data, Index
from brp.libs.lsh import _BRPModel, BucketedRandomProjection


class Main:
    @classmethod
    def run(cls, bucket_size: float, k: int, identifier: str) -> None:
        db_file: Path = Path(".databases/.index.sqlite3")
        index: Index = Index(db_file, force_init=False)
        brp: BucketedRandomProjection = BucketedRandomProjection(bucket_size=bucket_size)
        model: _BRPModel = brp.load_model(index, force_init=False)
        query: Data | None = index.fetch_data(identifier)

        if not query:
            print(f'No data found with identifier "{identifier}".')
            return None

        result = model.get_approximate_nearest_neighbors(query.embedding, k=k)
        pp(result)
        print("Total:", len(result))


if __name__ == "__main__":
    Main.run(float(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
