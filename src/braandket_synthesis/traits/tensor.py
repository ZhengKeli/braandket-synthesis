import abc
from typing import Optional, Union

from braandket import Backend, MixedStateTensor, OperatorTensor, PureStateTensor
from braandket_synthesis.basics import Op
from .apply import Apply, KetSpaces


class ToKraus(Apply[Op], abc.ABC):
    def __call__(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> tuple[OperatorTensor, ...]:
        return self.to_kraus(spaces, backend=backend)

    @abc.abstractmethod
    def to_kraus(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> tuple[OperatorTensor, ...]:
        pass

    def apply_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> tuple[Union[PureStateTensor, MixedStateTensor], None]:
        kraus_ops = self.to_kraus(spaces, backend=tensor.backend)

        # case when there is no Kraus operator, returning 0
        if len(kraus_ops) == 0:
            if isinstance(tensor, PureStateTensor):
                return PureStateTensor.of(0, ()), None
            elif isinstance(tensor, MixedStateTensor):
                return MixedStateTensor.of(0, ()), None

        # case when the resulting state can be pure (performing the tensor product)
        if len(kraus_ops) == 1:
            if isinstance(tensor, PureStateTensor):
                return kraus_ops[0] @ tensor, None

        # case when the resulting is mixed (performing the Kraus-sum)
        if isinstance(tensor, PureStateTensor):
            tensor = tensor @ tensor.ct
        return MixedStateTensor.of(sum(kop @ tensor @ kop.ct for kop in kraus_ops)), None


class ToTensor(ToKraus[Op], abc.ABC):
    def __call__(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        return self.to_tensor(spaces, backend=backend)

    @abc.abstractmethod
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        pass

    def to_kraus(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> tuple[OperatorTensor]:
        return self.to_tensor(spaces, backend=backend),
