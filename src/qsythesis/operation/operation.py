import abc
from typing import Generic, Iterable, Optional, TypeVar

SE = TypeVar('SE')
Op = TypeVar('Op', bound='QOperation')
Tr = TypeVar('Tr', bound='QOperationTrait')


# operation

class QOperation(abc.ABC, Generic[SE]):
    def __init__(self, *, name: Optional[str] = None):
        self._name = name

    @property
    def name(self) -> Optional[str]:
        return self._name

    # traits

    def trait(self, trait_cls: type[Tr], *, required: bool = True) -> Optional[Tr]:
        found_trait_cls = find_trait_cls(type(self), trait_cls)
        if found_trait_cls is None:
            if required:
                raise TypeError(f"Operation {self} does not support trait {trait_cls}")
            else:
                return None
        return found_trait_cls(self)

    # str & repr

    def __str__(self):
        return f"{type(self).__name__}"

    def __repr__(self):
        return f"<{type(self).__name__}>"


# trait

class QOperationTrait(Generic[Op], abc.ABC):
    def __init__(self, operation: Op):
        self._operation = operation

    @property
    def operation(self) -> Op:
        return self._operation


def find_trait_cls(op_cls: type, trait_cls: type) -> Optional[type]:
    picked_trait_cls = None
    picked_op_cls_order = None
    for t_cls in iter_subclasses_recursively(QOperationTrait):
        if not issubclass(t_cls, trait_cls):
            continue
        if getattr(t_cls, '__abstractmethods__', ()):
            continue
        for base in t_cls.__orig_bases__:  # type: ignore
            if not issubclass(base.__origin__, QOperationTrait):
                continue
            o_cls = base.__args__[0]
            if not issubclass(o_cls, op_cls):
                continue
            op_cls_order = op_cls.__mro__.index(o_cls)
            if picked_op_cls_order is not None and op_cls_order >= picked_op_cls_order:
                continue
            picked_trait_cls = t_cls
            picked_op_cls_order = op_cls_order
    return picked_trait_cls


def iter_subclasses_recursively(cls: type, iterated: Optional[set] = None) -> Iterable[type]:
    iterated = set() if iterated is None else iterated
    for sub_cls in cls.__subclasses__():
        if sub_cls not in iterated:
            yield sub_cls
            iterated.add(sub_cls)
        yield from iter_subclasses_recursively(sub_cls, iterated)
