from typing import Iterable

from braandket import MixedStateTensor, PureStateTensor
from braandket_synthesis import KetSpaces


def component_of_pure_state(
        tensor: PureStateTensor,
        spaces: Iterable[KetSpaces], values: Iterable[int],
) -> tuple[PureStateTensor, float]:
    ket_slices = tuple((space, value) for space, value in zip(spaces, values))
    component = PureStateTensor.of(tensor[ket_slices])

    prob = float(component.norm())
    component = component.normalize()
    return component, prob


def component_of_mixed_state(
        tensor: MixedStateTensor,
        spaces: Iterable[KetSpaces],
        values: Iterable[int]
) -> tuple[MixedStateTensor, float]:
    ket_slices = tuple((space, value) for space, value in zip(spaces, values))
    bra_slices = tuple((space.ct, value) for space, value in zip(spaces, values))
    component = MixedStateTensor.of(tensor[ket_slices + bra_slices])

    prob = float(component.norm())
    component = component.normalize()
    return component, prob
