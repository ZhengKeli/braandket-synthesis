import abc
from typing import Generic, Iterable, Union

from braandket import KetSpace, MixedStateTensor, PureStateTensor, QComposed, QModel, QParticle
from braandket_synthesis.basics import Op, QOperationTrait, SE

KetSpaces = Union[KetSpace, Iterable['KetSpaces']]


class Apply(QOperationTrait[Op], Generic[Op], abc.ABC):
    @abc.abstractmethod
    def apply_on_state_tensor(self,
            spaces: KetSpaces,
            tensor: Union[PureStateTensor, MixedStateTensor],
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], SE]:
        pass

    def apply_on_model(self, model: QModel) -> SE:
        assert isinstance(model, (QParticle, QComposed))

        state_tensor = model.state.tensor
        assert isinstance(state_tensor, (PureStateTensor, MixedStateTensor))

        state_tensor, side_effect = self.apply_on_state_tensor(model, state_tensor)
        model.state.tensor = state_tensor

        return side_effect
