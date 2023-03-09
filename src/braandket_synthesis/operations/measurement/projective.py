import itertools
from typing import Iterable, Union

import numpy as np

from braandket import MixedStateTensor, PureStateTensor, prod
from braandket_synthesis.basics import QOperation
from braandket_synthesis.traits import KetSpaces, Measure
from braandket_synthesis.utils import iter_structure, restore_structure
from .result import MeasurementResult


class ProjectiveMeasurement(QOperation):
    pass


class ProjectiveMeasurementMeasure(Measure[ProjectiveMeasurement, MeasurementResult[ProjectiveMeasurement]]):
    def measure_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], MeasurementResult[ProjectiveMeasurement]]:
        spaces = tuple(iter_structure(spaces))
        tensor, values, prob = projective_measure(tensor, spaces)
        values = restore_structure(values, spaces)
        return tensor, MeasurementResult(values=values, probability=prob, operation=self.operation)


def projective_measure(
        tensor: Union[PureStateTensor, MixedStateTensor],
        spaces: Iterable[KetSpaces],
) -> tuple[Union[PureStateTensor, MixedStateTensor], tuple[int, ...], float]:
    if isinstance(tensor, PureStateTensor):
        tensor, values, prob = projective_measure_on_pure_state(tensor, spaces)
    elif isinstance(tensor, MixedStateTensor):
        tensor, values, prob = projective_measure_on_mixed_state(tensor, spaces)
    else:
        raise TypeError(f"Unexpected type of tensor: {type(tensor)}")
    return tensor, values, prob


def projective_measure_on_pure_state(
        tensor: PureStateTensor,
        spaces: Iterable[KetSpaces]
) -> tuple[PureStateTensor, tuple[int, ...], float]:
    spaces = tuple(spaces)

    cases_values = tuple(itertools.product(*(range(space.n) for space in spaces)))
    cases_component = tuple(tensor.component(zip(spaces, case_values)) for case_values in cases_values)
    cases_prob = tuple(float(component.norm()) for component in cases_component)
    cases_component = tuple(component.normalize() for component in cases_component)

    cases_prob = np.asarray(cases_prob)
    cases_prob /= cases_prob.sum()

    case_i = np.random.choice(len(cases_values), p=cases_prob)
    values = cases_values[case_i]
    component = cases_component[case_i]
    prob = cases_prob[case_i]

    ket_tensor = PureStateTensor.of(prod(*(
        space.eigenstate(value, backend=tensor.backend)
        for space, value in zip(spaces, values)
    )))
    tensor = PureStateTensor.of(ket_tensor @ component)

    return tensor, values, prob


def projective_measure_on_mixed_state(
        tensor: MixedStateTensor,
        spaces: Iterable[KetSpaces]
) -> tuple[MixedStateTensor, tuple[int, ...], float]:
    spaces = tuple(spaces)

    cases_values = tuple(itertools.product(*(range(space.n) for space in spaces)))
    cases_component = tuple(tensor.component(zip(spaces, case_values)) for case_values in cases_values)
    cases_prob = tuple(float(component.norm()) for component in cases_component)
    cases_component = tuple(component.normalize() for component in cases_component)

    cases_prob = np.asarray(cases_prob)
    cases_prob /= cases_prob.sum()

    case_i = np.random.choice(len(cases_values), p=cases_prob)
    values = cases_values[case_i]
    component = cases_component[case_i]
    prob = cases_prob[case_i]

    ket_tensor = PureStateTensor.of(prod(*(
        space.eigenstate(value, backend=tensor.backend)
        for space, value in zip(spaces, values)
    )))
    tensor = MixedStateTensor.of(ket_tensor @ component @ ket_tensor.ct)

    return tensor, values, prob
