import abc
from typing import Optional

from braandket_synthesis import Op, QOperation, QOperationTrait


class IsDiagonal(QOperationTrait[Op], abc.ABC):
    def __call__(self) -> Optional[bool]:
        return self.is_diagonal()

    @abc.abstractmethod
    def is_diagonal(self) -> Optional[bool]:
        pass


class QOperationIsDiagonal(IsDiagonal[QOperation], abc.ABC):
    def is_diagonal(self) -> Optional[bool]:
        return None
