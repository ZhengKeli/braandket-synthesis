from typing import Optional

import numpy as np

from braandket import Backend, OperatorTensor
from braandket_synthesis.basics import QOperation
from braandket_synthesis.traits import KetSpaces, ToTensor
from braandket_synthesis.utils import iter_structure


class MatrixOperation(QOperation):
    def __init__(self, matrix: np.ndarray, *, name: Optional[str] = None):
        super().__init__(name=name)
        self._matrix = matrix

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class MatrixOperationToTensor(ToTensor[MatrixOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        spaces = tuple(iter_structure(spaces))
        return OperatorTensor.from_matrix(self.operation.matrix, spaces, backend=backend)
