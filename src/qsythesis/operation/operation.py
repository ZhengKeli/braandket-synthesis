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
    for tr_cls in iter_subclasses_recursively(QOperationTrait):
        if not issubclass(tr_cls, trait_cls):
            continue
        if getattr(tr_cls, '__abstractmethods__', ()):
            continue
        for tr_cls_base in tr_cls.__orig_bases__:  # type: ignore
            tr_cls_base_org = getattr(tr_cls_base, '__origin__', None)
            if tr_cls_base_org is None:
                continue
            if not issubclass(tr_cls_base_org, QOperationTrait):
                continue
            tr_cls_base_arg = tr_cls_base.__args__[0]
            if isinstance(tr_cls_base_arg, TypeVar):
                continue
            if not issubclass(op_cls, tr_cls_base_arg):
                continue
            op_cls_order = op_cls.__mro__.index(tr_cls_base_arg)
            if picked_op_cls_order is not None and op_cls_order >= picked_op_cls_order:
                continue
            picked_trait_cls = tr_cls
            picked_op_cls_order = op_cls_order
    return picked_trait_cls


def iter_subclasses_recursively(cls: type, iterated: Optional[set] = None) -> Iterable[type]:
    iterated = set() if iterated is None else iterated
    for sub_cls in cls.__subclasses__():
        if sub_cls not in iterated:
            yield sub_cls
            iterated.add(sub_cls)
        yield from iter_subclasses_recursively(sub_cls, iterated)
