from typing import Generic, Iterable, Optional

from braandket import Backend, OperatorTensor, prod
from braandket_synthesis.basics import Op, QOperation
from braandket_synthesis.traits import KetSpaces, ToTensor


class SequentialOperation(QOperation, Generic[Op]):
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
