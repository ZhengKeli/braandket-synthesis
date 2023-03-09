from typing import Iterable, Optional, Union

from braandket import MixedStateTensor, PureStateTensor, prod
from braandket_synthesis.basics import QOperation
from braandket_synthesis.traits import KetSpaces, Measure
from braandket_synthesis.utils import iter_structure
from .result import MeasurementResult


class DesiredMeasurement(QOperation):
    def __init__(self, values: Union[int, tuple], *, name: Optional[str] = None):
        super().__init__(name=name)
        self._values = values

    @property
    def values(self) -> Union[int, tuple]:
        return self._values


class DesiredMeasurementMeasure(Measure[DesiredMeasurement, MeasurementResult[DesiredMeasurement]]):
    def measure_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], MeasurementResult[DesiredMeasurement]]:
        spaces = tuple(iter_structure(spaces))
        values = tuple(iter_structure(self.operation.values))
        tensor, prob = desired_measure(tensor, spaces, values)
        return tensor, MeasurementResult(values=self.operation.values, probability=prob, operation=self.operation)


def desired_measure(
        tensor: Union[PureStateTensor, MixedStateTensor],
        spaces: Iterable[KetSpaces], values: Iterable[int],
) -> tuple[Union[PureStateTensor, MixedStateTensor], float]:
    if isinstance(tensor, PureStateTensor):
        tensor, prob = desired_measure_on_pure_state(tensor, spaces, values)
    elif isinstance(tensor, MixedStateTensor):
        tensor, prob = desired_measure_on_mixed_state(tensor, spaces, values)
    else:
        raise TypeError(f"Unexpected type of tensor: {type(tensor)}")
    return tensor, prob


def desired_measure_on_pure_state(
        tensor: PureStateTensor,
        spaces: Iterable[KetSpaces],
        values: Iterable[int],
) -> tuple[PureStateTensor, float]:
    spaces = tuple(spaces)
    values = tuple(values)

    component = tensor.component(((space, value) for space, value in zip(spaces, values)))
    prob = float(component.norm())
    component = component.normalize()

    ket_tensor = PureStateTensor.of(prod(*(
        space.eigenstate(value, backend=tensor.backend)
        for space, value in zip(spaces, values)
    )))
    tensor = PureStateTensor.of(ket_tensor @ component)

    return tensor, prob


def desired_measure_on_mixed_state(
        tensor: MixedStateTensor,
        spaces: Iterable[KetSpaces], values: Iterable[int],
) -> tuple[MixedStateTensor, float]:
    spaces = tuple(spaces)
    values = tuple(values)

    component = tensor.component(((space, value) for space, value in zip(spaces, values)))
    prob = float(component.norm())
    component = component.normalize()

    ket_tensor = PureStateTensor.of(prod(*(
        space.eigenstate(value, backend=tensor.backend)
        for space, value in zip(spaces, values)
    )))
    tensor = MixedStateTensor.of(ket_tensor @ component @ ket_tensor.ct)

    return tensor, prob
