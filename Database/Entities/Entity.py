from typing import Generic, Tuple
from typing_extensions import Self, TypeVarTuple, Unpack
from uuid import UUID, uuid4

# define the row format as a generic value
row_type = TypeVarTuple("row_type")

class Entity(Generic[Unpack[row_type]]):
    def __init__(self) -> None:
        self.id = uuid4()

    @staticmethod
    def from_row(id: str, *row: Unpack[row_type]):
        raise NotImplementedError("Error, please implement")

    def to_row(self) -> Tuple[str, Unpack[row_type]]:
        raise NotImplementedError("Error, please implement")

    def with_id(self, id: UUID) -> Self:
        self.id = id
        return self
