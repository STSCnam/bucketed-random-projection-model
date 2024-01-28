from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from brp.libs.types import Vector


@dataclass
class _BaseModel:
    def model_fields(self) -> dict[str, Any]:
        self_fields: dict[str, Any] = {}

        for f_name, f_obj in self.__dataclass_fields__.items():
            if f_obj.init:
                self_fields[f_name] = getattr(self, f_name)

        return self_fields

    def _prepare_mapping(self) -> dict[str, Any]:
        field_mapping: dict[str, Any] = {}

        for self_name, self_value in self.model_fields().items():
            if isinstance(self_value, _BaseModel):
                self_name = self_name + "_id"
                self_value = self_value.id
            if not isinstance(self_value, (float, int, str)) and self_value is not None:
                self_value = str(self_value)

            field_mapping[self_name] = self_value

        return field_mapping


@dataclass
class Bucket(_BaseModel):
    id: int = field(init=False, default=None)
    hash: int


@dataclass
class Data(_BaseModel):
    id: int = field(init=False, default=None)
    raw: Any
    embedding: Vector
    bucket: Bucket

    def __post_init__(self) -> None:
        if isinstance(self.embedding, str):
            self.embedding = tuple(map(float, self.embedding[1:-1].split(",")))


class Index:
    def __init__(self, db_file: Path, *, force_init: bool = False) -> None:
        self._db_file: Path = db_file
        self._db_conn: sqlite3.Connection = self._init_db(force_init)

    def fetch_bucket_data(self, bucket: Bucket) -> Iterator[Data]:
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

    def create(self, model: _BaseModel) -> _BaseModel | None:
        if isinstance(model, Data):
            return self._create_data(model)
        elif isinstance(model, Bucket):
            return self._create_bucket(model)

    def _create_data(self, model: Data) -> Data:
        if model.bucket.id is None:
            raise PermissionError(
                "Cannot create a Data entry without an existing Bucket."
            )

        f_mapping: dict[str, Any] = model._prepare_mapping()
        f_mapping["bucket_id"] = model.bucket.id
        model.id = self._execute_insert(model.__class__.__name__.lower(), **f_mapping)
        return model

    def _create_bucket(self, model: Bucket) -> Bucket:
        f_mapping: dict[str, Any] = model._prepare_mapping()
        model.id = self._execute_insert(model.__class__.__name__.lower(), **f_mapping)
        return model

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

    def _execute_select(self, t_name, **filter: Any) -> Iterator[Any]:
        stmt: str = (
            f"SELECT * FROM {t_name} WHERE "
            f"{' AND '.join(n + ' = ?' for n in filter)}"
        )
        cursor: sqlite3.Cursor = self._db_conn.cursor()
        cursor.execute(stmt, list(filter.values()))

        for row in cursor.fetchall():
            yield row

        cursor.close()

    def _init_db(self, force: bool) -> sqlite3.Connection:
        init_tables: bool = force or not self._db_file.exists()
        conn: sqlite3.Connection = sqlite3.connect(self._db_file)
        conn.row_factory = sqlite3.Row

        if init_tables:
            cursor: sqlite3.Cursor = conn.cursor()
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
                    embedding TEXT,
                    bucket_id INTEGER NOT NULL,
                    FOREIGN KEY (bucket_id) REFERENCES bucket (id)
                );
                """
            )
            cursor.close()

        return conn
