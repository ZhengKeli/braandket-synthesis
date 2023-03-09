from typing import Generic, Iterable, Optional, Union

from braandket import Backend, MixedStateTensor, OperatorTensor, PureStateTensor, prod, sum
from braandket_synthesis.basics import Op, QOperation
from braandket_synthesis.traits import Apply, KetSpaces, ToTensor
from braandket_synthesis.utils import iter_structure, iter_zip_structures


class Controlled(QOperation, Generic[Op]):
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


class ControlledApply(Apply[Controlled]):
    def apply_on_state_tensor(self,
            tensor: Union[PureStateTensor, MixedStateTensor],
            spaces: KetSpaces
    ) -> Union[PureStateTensor, MixedStateTensor]:
        control_spaces, target_spaces = spaces

        control_i_tensor = prod(*(
            sp.identity() for sp in iter_structure(control_spaces)))
        control_on_tensor = prod(*(
            sp.projector(k) for sp, k in iter_zip_structures(control_spaces, self.operation.keys)))
        control_off_tensor = control_i_tensor - control_on_tensor

        target_on_tensor = self.operation.bullet.trait(Apply).apply_on_state_tensor(tensor, target_spaces)
        target_off_tensor = tensor

        if isinstance(tensor, PureStateTensor):
            return PureStateTensor.of(sum(
                control_on_tensor @ target_on_tensor,
                control_off_tensor @ target_off_tensor
            ))
        elif isinstance(tensor, MixedStateTensor):
            return MixedStateTensor.of(sum(
                control_on_tensor @ target_on_tensor @ control_on_tensor,
                control_off_tensor @ target_off_tensor @ control_off_tensor
            ))
        else:
            raise TypeError(f"Expected type of tensor PureStateTensor or MixedStateTensor, got {type(tensor)}!")


class ControlledToTensor(ToTensor[Controlled]):
    def to_tensor(self, spaces: KetSpaces, *, backend: Optional[Backend] = None) -> OperatorTensor:
        control_spaces, target_spaces = spaces

        control_i_tensor = prod(*(
            sp.identity() for sp in iter_structure(control_spaces)))
        control_on_tensor = prod(*(
            sp.projector(k) for sp, k in iter_zip_structures(control_spaces, self.operation.keys)))
        control_off_tensor = control_i_tensor - control_on_tensor

        target_on_operator = self.operation.bullet.trait(ToTensor).to_tensor(target_spaces, backend=backend)
        target_off_operator = 1

        return OperatorTensor.of(sum(
            control_on_tensor @ target_on_operator,
            control_off_tensor @ target_off_operator
        ))
