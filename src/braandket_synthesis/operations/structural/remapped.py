from typing import Callable, Generic, Optional, Union

from braandket import Backend, MixedStateTensor, OperatorTensor, PureStateTensor
from braandket_synthesis.basics import Op, QOperation
from braandket_synthesis.traits import KetSpaces, Measure, R, ToTensor


class Remapped(QOperation, Generic[Op]):
    def __init__(self, original: Op, mapping: Callable[[KetSpaces], KetSpaces], *, name: Optional[str] = None):
        super().__init__(name=name)

        # check
        if not isinstance(original, QOperation):
            raise TypeError(f"original={original} is not a QOperation!")

        self._original = original
        self._mapping = mapping

    @property
    def original(self) -> Op:
        return self._original

    @property
    def mapping(self) -> Callable[[KetSpaces], KetSpaces]:
        return self._mapping


class RemappedMeasure(Measure[Remapped, R]):
    def measure_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], R]:
        return self.operation.trait(Measure).measure_on_state_tensor(tensor, self.operation.mapping(spaces))


class RemappedToTensor(ToTensor[Remapped]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.operation.original.trait(ToTensor).to_tensor(self.operation.mapping(spaces), backend=backend)
