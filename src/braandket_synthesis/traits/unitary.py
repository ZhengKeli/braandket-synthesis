import abc
from typing import Optional

from braandket_synthesis import Op, QOperation, QOperationTrait


class IsUnitary(QOperationTrait[Op], abc.ABC):
    def __call__(self) -> Optional[bool]:
        return self.is_unitary()

    @abc.abstractmethod
    def is_unitary(self) -> Optional[bool]:
        pass


class QOperationIsUnitary(IsUnitary[QOperation], abc.ABC):
    def is_unitary(self) -> Optional[bool]:
        return None
