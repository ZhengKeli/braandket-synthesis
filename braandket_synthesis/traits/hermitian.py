import abc
from typing import Optional

from braandket_synthesis import Op, QOperation, QOperationTrait


class IsHermitian(QOperationTrait[Op], abc.ABC):
    def __call__(self) -> Optional[bool]:
        return self.is_hermitian()

    @abc.abstractmethod
    def is_hermitian(self) -> Optional[bool]:
        pass


class QOperationIsHermitian(IsHermitian[QOperation], abc.ABC):
    def is_hermitian(self) -> Optional[bool]:
        return None
