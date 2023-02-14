import abc
from typing import Generic, Iterable, Optional, Union

from braandket import Backend, KetSpace, MixedStateTensor, OperatorTensor, PureStateTensor, QComposed, QModel, QParticle
from .operation import Op, QOperationTrait, SE

KetSpaces = Union[KetSpace, Iterable['KetSpaces']]


class ApplyOnPureState(QOperationTrait[Op], Generic[Op], abc.ABC):
    @abc.abstractmethod
    def apply_on_pure_state_tensor(self, spaces: KetSpaces, tensor: PureStateTensor) -> tuple[PureStateTensor, SE]:
        pass

    def apply_on_pure_state_model(self, model: QModel) -> SE:
        assert isinstance(model, (QParticle, QComposed))

        state_tensor = model.state.tensor
        assert isinstance(state_tensor, PureStateTensor)

        state_tensor, side_effect = self.apply_on_pure_state_tensor(model, state_tensor)

        model.state.tensor = state_tensor
        return side_effect


class ApplyOnMixedState(QOperationTrait[Op], Generic[Op], abc.ABC):
    @abc.abstractmethod
    def apply_on_mixed_state_tensor(self, spaces: KetSpaces, tensor: MixedStateTensor) -> tuple[MixedStateTensor, SE]:
        pass

    def apply_on_mixed_state_model(self, model: QModel) -> SE:
        assert isinstance(model, (QParticle, QComposed))

        state_tensor = model.state.tensor
        assert isinstance(state_tensor, MixedStateTensor)

        state_tensor, side_effect = self.apply_on_mixed_state_tensor(model, state_tensor)

        model.state.tensor = state_tensor
        return side_effect


class ToKraus(ApplyOnMixedState[Op], Generic[Op], abc.ABC):
    def __call__(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> tuple[OperatorTensor, ...]:
        return self.to_kraus(spaces, backend=backend)

    @abc.abstractmethod
    def to_kraus(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> tuple[OperatorTensor, ...]:
        pass

    def apply_on_mixed_state_tensor(self, spaces: KetSpaces, tensor: MixedStateTensor) -> tuple[MixedStateTensor, None]:
        return MixedStateTensor.of(sum(
            kop @ tensor @ kop.ct
            for kop in self.to_kraus(spaces, backend=tensor.backend)
        )), None


class ToTensor(ToKraus[Op], ApplyOnPureState[Op], Generic[Op], abc.ABC):
    def __call__(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.to_tensor(spaces, backend=backend)

    @abc.abstractmethod
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        pass

    def to_kraus(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> tuple[OperatorTensor, ...]:
        return self.to_tensor(spaces, backend=backend),

    def apply_on_pure_state_tensor(self, spaces: KetSpaces, tensor: PureStateTensor) -> tuple[PureStateTensor, None]:
        return self.to_tensor(spaces, backend=tensor.backend) @ tensor, None
