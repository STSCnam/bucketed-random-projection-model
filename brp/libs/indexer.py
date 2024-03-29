from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from brp.libs.types import Vector


@dataclass
class _BaseModel:
    """The base index models."""

    def model_fields(self) -> dict[str, Any]:
        """Get the available model fields.
        An field is considered available if its init property is True.

        Returns:
            dict[str, Any]: The model's fields mapping (key, value).
        """

        self_fields: dict[str, Any] = {}

        for f_name, f_obj in self.__dataclass_fields__.items():
            if f_obj.init:
                self_fields[f_name] = getattr(self, f_name)

        return self_fields

    def _prepare_mapping(self) -> dict[str, Any]:
        """Fit the model's (key, value) mapping to the index model.

        Returns:
            dict[str, Any]: The transformed model's field mapping.
        """

        field_mapping: dict[str, Any] = {}

        for self_name, self_value in self.model_fields().items():
            if isinstance(self_value, _BaseModel):
                self_name = self_name + "_id"
                self_value = self_value.id
            if not isinstance(self_value, (float, int, str)) and self_value is not None:
                self_value = str(self_value)
            if self_value is not None:
                field_mapping[self_name] = self_value

        return field_mapping


@dataclass
class Hyperplane(_BaseModel):
    """The hyperplane model.

    Attributes:
        id: The hyperplane's index.
        vector: The hyperplane point.
    """

    id: int = field(init=False, default=None)
    vector: Vector

    def __post_init__(self) -> None:
        if isinstance(self.vector, str):
            self.vector = tuple(map(float, self.vector[1:-1].split(",")))


@dataclass
class Bucket(_BaseModel):
    """The bucket model.

    Attributes:
        id: The bucket's index.
        hash: The bucket's hash.
    """

    id: int = field(init=False, default=None)
    hash: int


@dataclass
class Data(_BaseModel):
    """The data model.

    Attributes:
        id: The data's index.
        raw: The raw information that identify the data (for tests).
        embedding: The data embedding.
        bucket: The data's related bucket.
    """

    id: int = field(init=False, default=None)
    raw: Any
    embedding: Vector
    bucket: Bucket | None = None

    def __post_init__(self) -> None:
        if isinstance(self.embedding, str):
            self.embedding = tuple(map(float, self.embedding[1:-1].split(",")))


class Index:
    """The index manager class that will create, update and fetch the dataset's index."""

    def __init__(self, db_file: Path, *, force_init: bool = False) -> None:
        """The index initializer.

        Args:
            db_file: The SQLite3 file that contains the index.
            force_init (Optional): Force the index creation from scratch.
                Default to False.
        """

        self._db_conn: sqlite3.Connection = self._init_db(db_file, force_init)

    def populate(self, dataset: list[dict[str, Any]]) -> None:
        """Populate the index from the dataset object.

        Args:
            dataset: The dataset to import.
        """

        for data in dataset:
            self.create(Data(**data))

    def clean(self, *model_type: type[_BaseModel]) -> None:
        """Truncate the index tables according to the given model types.

        Args:
            model_type: The multiple argument of the model types to clean.
        """

        cursor: sqlite3.Cursor = self._db_conn.cursor()

        for m_type in model_type:
            cursor.execute(f"DELETE FROM {m_type.__name__.lower()}")

        self._db_conn.commit()
        cursor.close()

    def fetch_all_data(self) -> Iterator[Data]:
        """Get all data from index.

        Returns:
            Iterator[Data]: The fetched data iterator.
        """

        for row in self._execute_select(Data.__name__.lower()):
            rel_bucket: Bucket | None = None
            row = dict(row)
            cached_id: int = row["id"]

            if row["bucket_id"] is not None:
                rel_bucket = self.fetch_bucket(row["bucket_id"])

            del row["id"], row["bucket_id"]
            data: Data = Data(**row)
            data.id = cached_id
            data.bucket = rel_bucket
            yield data

    def fetch_all_hyperplanes(self) -> Iterator[Hyperplane]:
        """Get all hyperplanes from index.

        Returns:
            Iterator[Hyperplane]: An iterator of the fetched hyperplanes.
        """

        for row in self._execute_select(Hyperplane.__name__.lower()):
            row = dict(row)
            cached_id: int = row["id"]
            del row["id"]
            hyperplane: Hyperplane = Hyperplane(**row)
            hyperplane.id = cached_id
            yield hyperplane

    def fetch_bucket(self, hash: int) -> Bucket | None:
        """Get the bucket from index by the given hash.

        Args:
            hash: The bucket's hash to fetch.

        Returns:
            Bucket | None: The fetched bucket if exists, else None.
        """

        row: dict[str, Any] | None = None

        for row in self._execute_select(Bucket.__name__.lower(), hash=hash):
            row = dict(row)
            break

        if row is not None:
            cached_id: int = row["id"]
            del row["id"]
            bucket: Bucket = Bucket(**row)
            bucket.id = cached_id
            return bucket

    def fetch_data(self, raw: str) -> Data | None:
        """Get the data from index by its identifier.
        The data if returned with its related bucket if exists.

        Args:
            raw: The data identifier.

        Returns:
            Data | None: The fetched data, else None.
        """

        row: dict[str, Any] | None = None

        for row in self._execute_select(Data.__name__.lower(), raw=raw):
            row = dict(row)
            break

        if row:
            rel_bucket: Bucket | None = None
            row = dict(row)
            cached_id: int = row["id"]

            if row["bucket_id"] is not None:
                rel_bucket = self.fetch_bucket(row["bucket_id"])

            del row["id"], row["bucket_id"]
            data: Data = Data(**row)
            data.id = cached_id
            data.bucket = rel_bucket
            return data

    def fetch_bucket_data(self, bucket: Bucket) -> Iterator[Data]:
        """Get the data set related to the given bucket.

        Args:
            bucket: The bucket model that contains its related data set.

        Returns:
            Iterator[Data]: The bucket's related data set if any.
        """

        if bucket.id is None:
            raise PermissionError("Cannot get bucket's data if it not exists.")

        for row in self._execute_select(Data.__name__.lower(), bucket_id=bucket.id):
            row = dict(row)
            cached_id: int = row["id"]
            row["bucket"] = bucket
            del row["id"], row["bucket_id"]
            data: Data = Data(**row)
            data.id = cached_id
            yield data

    def update(self, model: _BaseModel) -> _BaseModel:
        """Update index.

        Args:
            model: The Data or Bucket model to update.

        Raises:
            TypeError: If the given model is not one of (Data, Bucket) objects.

        Returns:
            _BaseModel | None: The updated data.
        """

        if isinstance(model, Data):
            return self._update_data(model)
        elif isinstance(model, Bucket):
            return self._update_bucket(model)
        else:
            raise TypeError(
                f"Update not permitted for model {model.__class__.__name__}."
            )

    def create(self, model: _BaseModel) -> _BaseModel | None:
        """Create an entry to the index according to the given model.

        Args:
            model: The model to create.

        Returns:
            _BaseModel | None: The created model if any.
        """

        if isinstance(model, Data):
            return self._create_data(model)
        elif isinstance(model, Bucket):
            return self._create_bucket(model)
        elif isinstance(model, Hyperplane):
            return self._create_hyperplane(model)

    def _create_hyperplane(self, model: Hyperplane) -> Hyperplane:
        f_mapping: dict[str, Any] = model._prepare_mapping()
        self._execute_insert(model.__class__.__name__.lower(), **f_mapping)
        return model

    def _update_data(self, model: Data) -> Data:
        f_mapping: dict[str, Any] = model._prepare_mapping()

        if model.bucket:
            f_mapping["bucket_id"] = model.bucket.id

        self._execute_update(model.__class__.__name__.lower(), model.id, **f_mapping)
        return model

    def _update_bucket(self, model: Bucket) -> Bucket:
        f_mapping: dict[str, Any] = model._prepare_mapping()
        self._execute_update(model.__class__.__name__.lower(), model.id, **f_mapping)
        return model

    def _create_data(self, model: Data) -> Data:
        f_mapping: dict[str, Any] = model._prepare_mapping()

        if model.bucket:
            f_mapping["bucket_id"] = model.bucket.id

        model.id = self._execute_insert(model.__class__.__name__.lower(), **f_mapping)
        return model

    def _create_bucket(self, model: Bucket) -> Bucket:
        if fetched_model := self.fetch_bucket(model.hash):
            return fetched_model

        f_mapping: dict[str, Any] = model._prepare_mapping()
        model.id = self._execute_insert(model.__class__.__name__.lower(), **f_mapping)
        return model

    def _execute_update(self, t_name: str, id_: int, **f_mapping: Any) -> None:
        stmt: str = (
            f"UPDATE {t_name} "
            f"SET {', '.join(n + ' = ?' for n in f_mapping)} "
            f"WHERE id = ?"
        )
        f_mapping["id"] = id_
        cursor: sqlite3.Cursor = self._db_conn.cursor()
        cursor.execute(stmt, list(f_mapping.values()))
        self._db_conn.commit()
        cursor.close()

    def _execute_insert(self, t_name: str, **f_mapping: Any) -> int:
        stmt: str = (
            f"INSERT INTO {t_name} "
            f"({', '.join(f_mapping)}) VALUES "
            f"({', '.join('?' for _ in range(len(f_mapping)))})"
        )
        cursor: sqlite3.Cursor = self._db_conn.cursor()
        cursor.execute(stmt, list(f_mapping.values()))
        self._db_conn.commit()
        id_: int = cursor.lastrowid
        cursor.close()
        return id_

    def _execute_select(self, t_name, **filters: Any) -> Iterator[Any]:
        stmt: str = f"SELECT * FROM {t_name}"

        if len(filters):
            stmt += f" WHERE {' AND '.join(n + ' = ?' for n in filters)}"

        cursor: sqlite3.Cursor = self._db_conn.cursor()
        cursor.execute(stmt, list(filters.values()))

        for row in cursor.fetchall():
            yield row

        cursor.close()

    def _init_db(self, db_file: Path, force: bool) -> sqlite3.Connection:
        init_tables: bool = force or not db_file.exists()

        if init_tables:
            db_file.unlink(missing_ok=True)

        conn: sqlite3.Connection = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row

        if init_tables:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """\
                CREATE TABLE IF NOT EXISTS hyperplane (
                    id INTEGER PRIMARY KEY,
                    vector TEXT NOT NULL
                );
                """
            )
            cursor.execute(
                """\
                CREATE TABLE IF NOT EXISTS bucket (
                    id INTEGER PRIMARY KEY,
                    hash INTEGER NOT NULL
                );
                """
            )
            cursor.execute("PRAGMA foreign_key = ON;")
            cursor.execute(
                """\
                CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY,
                    raw TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    bucket_id INTEGER,
                    FOREIGN KEY (bucket_id) REFERENCES bucket (id)
                );
                """
            )
            cursor.close()

        return conn
