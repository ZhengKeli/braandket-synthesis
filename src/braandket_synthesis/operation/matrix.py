from typing import Optional

import numpy as np

from braandket import Backend, OperatorTensor
from .operation import QOperation
from .traits import KetSpaces, ToTensor
from .utils import iter_structured


# generic

class MatrixOperation(QOperation):
    def __init__(self, matrix: np.ndarray, *, name: Optional[str] = None):
        super().__init__(name=name)
        self._matrix = matrix

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class MatrixOperationToTensor(ToTensor[MatrixOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        spaces = tuple(iter_structured(spaces))
        return OperatorTensor.from_matrix(self.operation.matrix, spaces, backend=backend)


# qubits

class QubitsMatrixOperation(MatrixOperation):
    def __init__(self, matrix: np.ndarray, *, name: Optional[str] = None):
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"expected matrix shape (2**n, 2**n), got {shape}")
        N = shape[0]
        n = log2int(N, strict=True)
        if n is None:
            raise ValueError(f"expected matrix shape (2**n, 2**n), got {shape}")
        super().__init__(matrix, name=name)
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    @property
    def N(self) -> int:
        return 2 ** self.n


def log2int(x: int, *, strict: bool = False) -> Optional[int]:
    if strict and x & (x - 1) != 0:
        return None
    return x.bit_length() - 1
