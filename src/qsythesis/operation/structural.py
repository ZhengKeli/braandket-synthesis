from typing import Callable, Generic, Iterable, Optional, Union

from braandket.backend import Backend
from braandket.tensor import OperatorTensor, prod, sum
from .operation import Op, QOperation, SE
from .traits import KetSpaces, TensorTrait
from .utils import iter_structured, iter_structured_zip


# sequential

class SequentialOperation(QOperation[tuple], Generic[Op]):
    def __init__(self, steps: Iterable[Op], *, name: Optional[str] = None):
        super().__init__(name=name)
        self._steps = tuple(steps)

    @property
    def steps(self) -> tuple[Op, ...]:
        return self._steps


class SequentialOperationTensorTrait(TensorTrait[SequentialOperation]):
    def tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return OperatorTensor.of(prod(*(
            step.trait(TensorTrait).tensor(spaces, backend=backend)
            for step in self.operation.steps
        )))


# remapped

class RemappedOperation(QOperation[SE], Generic[Op, SE]):
    def __init__(self, operation: Op, mapping: Callable[[KetSpaces], KetSpaces], *, name: Optional[str] = None):
        super().__init__(name=name)
        self._operation = operation
        self._mapping = mapping

    @property
    def operation(self) -> Op:
        return self._operation

    @property
    def mapping(self) -> Callable[[KetSpaces], KetSpaces]:
        return self._mapping


class RemappedOperationTensorTrait(TensorTrait[RemappedOperation]):
    def tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.operation.operation.trait(TensorTrait).tensor(self.operation.mapping(spaces), backend=backend)


# controlled

class ControlledOperation(QOperation[SE], Generic[Op, SE]):
    def __init__(self, operation: Op, keys: Union[int, Iterable] = 1, *, name: Optional[str] = None):
        super().__init__(name=name)
        self._operation = operation
        self._keys = keys

    @property
    def operation(self) -> Op:
        return self._operation

    @property
    def keys(self) -> Union[int, Iterable]:
        return self._keys


class ControlledOperationTensorTrait(TensorTrait[ControlledOperation]):
    def tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        control_spaces, target_spaces = spaces

        control_i_tensor = prod(*(
            sp.identity() for sp in iter_structured(control_spaces)))
        control_on_tensor = prod(*(
            sp.projector(k) for sp, k in iter_structured_zip(control_spaces, self.operation.keys)))
        control_off_tensor = control_i_tensor - control_on_tensor

        target_on_operator = self.operation.operation.trait(TensorTrait).tensor(target_spaces, backend=backend)
        target_off_operator = 1

        return OperatorTensor.of(sum(
            control_on_tensor @ target_on_operator,
            control_off_tensor @ target_off_operator
        ))
