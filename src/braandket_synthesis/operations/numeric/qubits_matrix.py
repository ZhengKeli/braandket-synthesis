from typing import Optional

import numpy as np

from .matrix import MatrixOperation


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
