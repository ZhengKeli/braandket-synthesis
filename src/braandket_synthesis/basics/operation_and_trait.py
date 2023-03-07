import abc
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

Op = TypeVar('Op', bound='QOperation')
Tr = TypeVar('Tr', bound='QOperationTrait')


# operation

class QOperation(abc.ABC):
    def __init__(self, *, name: Optional[str] = None):
        self._name = name
        self._cache: dict[type[Tr], dict[str, Any]] = {}

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

    # cache

    NoDefault = object()

    def get_cache(self, trait_cls: type[Tr], key: str, default: Any = NoDefault) -> Any:
        trait_cls_order = None
        trait_cache_value = default
        for cache_trait_cls, cache_trait_dict in self._cache.items():
            if not issubclass(trait_cls, cache_trait_cls):
                continue
            if key not in cache_trait_dict:
                continue
            cache_trait_cls_order = trait_cls.mro().index(cache_trait_cls)
            if trait_cls_order is None or cache_trait_cls_order < trait_cls_order:
                trait_cls_order = cache_trait_cls_order
                trait_cache_value = cache_trait_dict[key]
        if trait_cache_value is self.NoDefault:
            raise KeyError(f"Not found cache value for {trait_cls} and key {key}")
        return trait_cache_value

    def set_cache(self, trait_cls: type[Tr], key: str, value: Any):
        cache_trait_dict = self._cache.get(trait_cls)
        if cache_trait_dict is None:
            cache_trait_dict = {}
            self._cache[trait_cls] = cache_trait_dict
        cache_trait_dict[key] = value

    def get_or_set_cache(self, trait_cls: type[Tr], key: str, default_func: Callable[[], Any]) -> Any:
        try:
            return self.get_cache(trait_cls, key)
        except KeyError:
            value = default_func()
            self.set_cache(trait_cls, key, value)
            return value


# trait

class QOperationTrait(Generic[Op], abc.ABC):
    def __init__(self, operation: Op):
        self._operation = operation

    @property
    def operation(self) -> Op:
        return self._operation

    # cache

    def get_cache(self, key: str, default: Any = QOperation.NoDefault) -> Any:
        return self.operation.get_cache(type(self), key, default)

    def set_cache(self, key: str, value: Any):
        self.operation.set_cache(type(self), key, value)

    def get_or_set_cache(self, key: str, default_func: Callable[[], Any]) -> Any:
        return self.operation.get_or_set_cache(type(self), key, default_func)

    # operation class resolving

    @classmethod
    def op_cls_list(cls) -> tuple[type[Op], ...]:
        op_cls_list = []
        for tr_cls_base in getattr(cls, '__orig_bases__', ()):
            tr_cls_base_org = getattr(tr_cls_base, '__origin__', None)
            if tr_cls_base_org is None:
                continue
            if not issubclass(tr_cls_base_org, QOperationTrait):
                continue
            tr_cls_base_args = getattr(tr_cls_base, '__args__', ())
            if len(tr_cls_base_args) == 0:
                raise TypeError(f"{cls} is expected to have a subclass of QOperation as the first type argument!"
                                f"Got no type arguments!")
            tr_cls_base_arg = tr_cls_base_args[0]
            if isinstance(tr_cls_base_arg, TypeVar):
                continue
            if not issubclass(tr_cls_base_arg, QOperation):
                raise TypeError(f"{cls} is expected to have a subclass of QOperation as the first type argument!"
                                f"Got {tr_cls_base_arg}!")
            op_cls_list.append(tr_cls_base_arg)
        return tuple(op_cls_list)

    @classmethod
    def op_cls_order(cls, op_cls: type[Op]) -> Optional[int]:
        resolved_op_cls_order = None
        for trait_op_cls in cls.op_cls_list():
            try:
                op_cls_order = op_cls.mro().index(trait_op_cls)
            except ValueError:
                continue
            if resolved_op_cls_order is not None and op_cls_order >= resolved_op_cls_order:
                continue
            resolved_op_cls_order = op_cls_order
        return resolved_op_cls_order


def find_trait_cls(op_cls: type[Op], trait_cls: type[Tr]) -> Optional[type[Tr]]:
    resolved = None
    for tr_cls, tr_cls_order, op_cls_order in iter_trait_cls(op_cls, trait_cls):
        if resolved is not None:
            resolved_tr_cls, resolved_tr_cls_order, resolved_op_cls_order = resolved
            if op_cls_order > resolved_op_cls_order:
                continue
            if op_cls_order == resolved_op_cls_order:
                if tr_cls_order >= resolved_tr_cls_order:
                    continue
        resolved = tr_cls, tr_cls_order, op_cls_order
    return resolved[0] if resolved is not None else None


def iter_trait_cls(op_cls: type[Op], trait_cls: type[Tr]) -> Optional[type[Tr]]:
    for tr_cls in iter_subclasses_recursively(QOperationTrait, reverse=True):
        if not issubclass(tr_cls, trait_cls):
            continue
        if getattr(tr_cls, '__abstractmethods__', ()):
            continue
        tr_cls_order = tr_cls.mro().index(trait_cls)
        op_cls_order = tr_cls.op_cls_order(op_cls)
        if op_cls_order is None:
            continue
        yield tr_cls, tr_cls_order, op_cls_order


def iter_subclasses_recursively(cls: type, iterated: Optional[set] = None, reverse: bool = False) -> Iterable[type]:
    iterated = set() if iterated is None else iterated
    subclasses = cls.__subclasses__()
    subclasses = reversed(subclasses) if reverse else subclasses
    for sub_cls in subclasses:
        if sub_cls not in iterated:
            yield sub_cls
            iterated.add(sub_cls)
        yield from iter_subclasses_recursively(sub_cls, iterated)
