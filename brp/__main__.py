from pathlib import Path
from brp.libs.indexer import Bucket, Data, Index


class Main:
    def run() -> None:
        index: Index = Index(Path(".databases/.index.sqlite3"))
        b = Bucket(hash=2)
        b2 = Bucket(hash=4)
        d = Data(
            raw="hello",
            embedding=(1.12, 2.45),
            bucket=b,
        )
        d2 = Data(
            raw="hello",
            embedding=(4.45861, 2.45),
            bucket=b,
        )
        d3 = Data(
            raw="hello",
            embedding=(4.45861, 2.45),
            bucket=b,
        )
        index.create(b)
        index.create(b2)
        index.create(d)
        index.create(d2)
        index.create(d3)
        print(*index.fetch_bucket_data(b))
        print(*index.fetch_bucket_data(b2))


if __name__ == "__main__":
    Main.run()
