from typing import Callable, Generic, Iterable, Optional, Union

from braandket.backend import Backend
from braandket.tensor import OperatorTensor, prod, sum
from braandket_synthesis.basics import Op, QOperation, SE
from braandket_synthesis.traits import KetSpaces, ToTensor
from .utils import iter_structured, iter_structured_zip


# sequential

class SequentialOperation(QOperation[tuple], Generic[Op]):
    def __init__(self, steps: Iterable[Op], *, name: Optional[str] = None):
        super().__init__(name=name)

        # check
        steps = tuple(steps)
        for i, step in enumerate(steps):
            if not isinstance(step, QOperation):
                raise TypeError(f"steps[{i}]={step} is not a QOperation!")

        self._steps = steps

    @property
    def steps(self) -> tuple[Op, ...]:
        return self._steps


class SequentialOperationToTensor(ToTensor[SequentialOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return OperatorTensor.of(prod(*(
            step.trait(ToTensor).to_tensor(spaces, backend=backend)
            for step in self.operation.steps
        )))


# remapped

class RemappedOperation(QOperation[SE], Generic[Op, SE]):
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


class RemappedOperationToTensor(ToTensor[RemappedOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.operation.original.trait(ToTensor).to_tensor(self.operation.mapping(spaces), backend=backend)


# controlled

class ControlledOperation(QOperation[SE], Generic[Op, SE]):
    def __init__(self, bullet: Op, keys: Union[int, Iterable] = 1, *, name: Optional[str] = None):
        super().__init__(name=name)

        # check
        if not isinstance(bullet, QOperation):
            raise TypeError(f"bullet={bullet} is not a QOperation!")

        self._bullet = bullet
        self._keys = keys

    @property
    def bullet(self) -> Op:
        return self._bullet

    @property
    def keys(self) -> Union[int, Iterable]:
        return self._keys


class ControlledOperationToTensor(ToTensor[ControlledOperation]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        control_spaces, target_spaces = spaces

        control_i_tensor = prod(*(
            sp.identity() for sp in iter_structured(control_spaces)))
        control_on_tensor = prod(*(
            sp.projector(k) for sp, k in iter_structured_zip(control_spaces, self.operation.keys)))
        control_off_tensor = control_i_tensor - control_on_tensor

        target_on_operator = self.operation.bullet.trait(ToTensor).to_tensor(target_spaces, backend=backend)
        target_off_operator = 1

        return OperatorTensor.of(sum(
            control_on_tensor @ target_on_operator,
            control_off_tensor @ target_off_operator
        ))
