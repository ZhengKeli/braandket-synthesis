import abc
from typing import Generic, TypeVar, Union

from braandket import MixedStateTensor, PureStateTensor, QComposed, QModel, QParticle
from braandket_synthesis.basics import QOperation
from .apply import Apply, KetSpaces, Op

R = TypeVar('R')


class Measure(Apply[Op], Generic[Op, R], abc.ABC):
    @abc.abstractmethod
    def measure_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], R]:
        pass

    def measure_on_model(self, model: QModel) -> R:
        assert isinstance(model, (QParticle, QComposed))

        state_tensor = model.state.tensor
        assert isinstance(state_tensor, (PureStateTensor, MixedStateTensor))

        state_tensor, result = self.measure_on_state_tensor(state_tensor, model)
        model.state.tensor = state_tensor

        return result

    def apply_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> Union[PureStateTensor, MixedStateTensor]:
        tensor, result = self.measure_on_state_tensor(tensor, spaces)
        return tensor

    def apply_on_model(self, model: QModel):
        self.measure_on_model(model)


class QOperationMeasure(Measure[QOperation, None]):
    def measure_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], None]:
        tensor = self.operation.trait(Apply).apply_on_state_tensor(tensor, spaces)
        return tensor, None
