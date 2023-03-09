import abc
from typing import Iterable, Union

from braandket import KetSpace, MixedStateTensor, PureStateTensor, QComposed, QModel, QParticle
from braandket_synthesis.basics import Op, QOperationTrait

KetSpaces = Union[KetSpace, Iterable['KetSpaces']]


class Apply(QOperationTrait[Op], abc.ABC):
    @abc.abstractmethod
    def apply_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> Union[PureStateTensor, MixedStateTensor]:
        pass

    def apply_on_model(self, model: QModel):
        assert isinstance(model, (QParticle, QComposed))

        state_tensor = model.state.tensor
        assert isinstance(state_tensor, (PureStateTensor, MixedStateTensor))

        state_tensor = self.apply_on_state_tensor(state_tensor, model)
        model.state.tensor = state_tensor
