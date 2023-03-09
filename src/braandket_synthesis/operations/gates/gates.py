import abc
from typing import Any, Optional, Union

import numpy as np

from braandket_synthesis.operations.measurement import DesiredMeasurement, ProjectiveMeasurement
from braandket_synthesis.operations.numeric import QubitsMatrixOperation
from braandket_synthesis.operations.structural import Controlled


# abstract

class QuantumGate(QubitsMatrixOperation):
    pass


class QuantumGateWithParam(QuantumGate, abc.ABC):
    @property
    @abc.abstractmethod
    def arguments(self) -> dict[str, Any]:
        pass


# single qubit gates

class X(QuantumGate):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(np.asarray([
            [0, 1],
            [1, 0]
        ]), name=name)


class Y(QuantumGate):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(np.asarray([
            [0, -1j],
            [+1j, 0]
        ]), name=name)


class Z(QuantumGate):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(np.asarray([
            [1, 0],
            [0, -1]
        ]), name=name)


class S(QuantumGate):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(np.asarray([
            [1, 0],
            [0, 1j]
        ]), name=name)


class T(QuantumGate):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(np.asarray([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ]), name=name)


class H(QuantumGate):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(np.asarray([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2), name=name)


NOT = X


# single qubit gates with parameter

class Rx(QuantumGateWithParam):
    def __init__(self, theta: Union[np.ndarray, float], *, name: Optional[str] = None):
        self._theta = theta
        half_theta = self.theta / 2
        super().__init__(np.asarray([
            [np.cos(half_theta) * +1, np.sin(half_theta) * -1j],
            [np.sin(half_theta) * -1j, np.cos(half_theta) * +1]
        ]), name=name)

    @property
    def theta(self) -> Union[np.ndarray, float]:
        return self._theta

    @property
    def arguments(self) -> dict[str, Any]:
        return {'theta': self.theta}


class Ry(QuantumGateWithParam):
    def __init__(self, theta: Union[np.ndarray, float], *, name: Optional[str] = None):
        self._theta = theta
        half_theta = self.theta / 2
        super().__init__(np.asarray([
            [np.cos(half_theta) * +1, np.sin(half_theta) * -1],
            [np.sin(half_theta) * +1, np.cos(half_theta) * +1]
        ]), name=name)

    @property
    def theta(self) -> Union[np.ndarray, float]:
        return self._theta

    @property
    def arguments(self) -> dict[str, Any]:
        return {'theta': self.theta}


class Rz(QuantumGateWithParam):
    def __init__(self, theta: Union[np.ndarray, float], *, name: Optional[str] = None):
        self._theta = theta
        half_theta_j = self.theta / 2 * 1j
        super().__init__(np.asarray([
            [np.exp(-half_theta_j), 0],
            [0, np.exp(+half_theta_j)]
        ]), name=name)

    @property
    def theta(self) -> Union[np.ndarray, float]:
        return self._theta

    @property
    def arguments(self) -> dict[str, Any]:
        return {'theta': self.theta}


# controlled qubit gates

class CX(Controlled[X]):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(X(), 1, name=name)


class CY(Controlled[Y]):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(Y(), 1, name=name)


class CZ(Controlled[Z]):
    def __init__(self, *, name: Optional[str] = None):
        super().__init__(Z(), 1, name=name)


CNOT = CX

# measurements

D = DesiredMeasurement

M = ProjectiveMeasurement
