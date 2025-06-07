

import traceback
from typing import Any, Generic, Type, TypeVar
from uuid import UUID
from Database.Entities.Entity import Entity
from Database.dbhandler import DbConnection

T = TypeVar('T', bound=Entity)

class Repository(Generic[T]):
    _class: Type[T]
    def __init__(self, db: DbConnection):
        self.db = db

    def all(self) -> list[T]:
        rows = self.db.execute_get(f"SELECT * FROM {self.table_name()}")
        result = []
        for row in rows:
            try:
                result.append(self._class.from_row(*row))
            except Exception as e:
                traceback.print_exception(e)
                print("Failed to load row, continuing")
        return result
    
    def get_by(self, name: str, value: Any) -> list[T]:
        rows = self.db.execute_get(f"SELECT * FROM {self.table_name()} WHERE {name} = %s", [
            value
        ])
        return [
            self._class.from_row(*row)
            for row in rows
        ]
    
    def get_by_id(self, id: UUID) -> T:
        rows = self.get_by("id", str(id))
        if len(rows) == 0:
            raise ValueError(f"No service with {id=}")
        return rows[0]


    def delete_all(self):
        self.db.execute(f"DELETE FROM {self.table_name()}")

    def insert(self, entity: T):
        row = entity.to_row()
        self.db.execute(f"INSERT INTO {self.table_name()} VALUES ({'%s'*len(row)})", row)

    def table_name(self) -> str:
        return self._class.__name__.lower()
