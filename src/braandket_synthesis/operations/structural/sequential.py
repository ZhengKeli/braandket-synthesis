from typing import Generic, Iterable, Optional, Union

from braandket import Backend, MixedStateTensor, OperatorTensor, PureStateTensor, prod
from braandket_synthesis.basics import Op, QOperation
from braandket_synthesis.traits import KetSpaces, Measure, ToTensor


class Sequential(QOperation, Generic[Op]):
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


class SequentialMeasure(Measure[Sequential, tuple]):
    def measure_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], tuple]:
        results = []
        for step in self.operation.steps:
            tensor, result = step.trait(Measure).measure_on_state_tensor(tensor, spaces)
            results.append(result)
        return tensor, tuple(results)


class SequentialToTensor(ToTensor[Sequential]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return OperatorTensor.of(prod(*(
            step.trait(ToTensor).to_tensor(spaces, backend=backend)
            for step in self.operation.steps
        )))
