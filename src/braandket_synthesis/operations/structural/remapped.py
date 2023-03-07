from typing import Callable, Generic, Optional, Union

from braandket import Backend, MixedStateTensor, OperatorTensor, PureStateTensor
from braandket_synthesis.basics import Op, QOperation
from braandket_synthesis.traits import KetSpaces, Measure, R, ToTensor


class RemappedOperation(QOperation, Generic[Op]):
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


class RemappedOperationMeasure(Measure[RemappedOperation, R]):
    def measure_on_state_tensor(self,
            spaces: KetSpaces,
            tensor: Union[PureStateTensor, MixedStateTensor]
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], R]:
        return self.operation.trait(Measure).measure_on_state_tensor(self.operation.mapping(spaces), tensor)


class RemappedOperationToTensor(ToTensor[RemappedOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.operation.original.trait(ToTensor).to_tensor(self.operation.mapping(spaces), backend=backend)
