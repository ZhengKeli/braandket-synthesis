from typing import Callable, Generic, Optional

from braandket import Backend, OperatorTensor
from braandket_synthesis.basics import Op, QOperation, SE
from braandket_synthesis.traits import KetSpaces, ToTensor


class RemappedOperation(QOperation[SE], Generic[Op, SE]):
    def __init__(self, original: Op, mapping: Callable[[KetSpaces], KetSpaces], *, name: Optional[str] = None):
        super().__init__(name=name)

        # check
        if not isinstance(original, QOperation):
            raise TypeError(f"original={original} is not a QOperation!")

        self._original = original
        self._mapping = mapping

    @property
    def original(self) -> Op:
        return self._original

    @property
    def mapping(self) -> Callable[[KetSpaces], KetSpaces]:
        return self._mapping


class RemappedOperationToTensor(ToTensor[RemappedOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.operation.original.trait(ToTensor).to_tensor(self.operation.mapping(spaces), backend=backend)
