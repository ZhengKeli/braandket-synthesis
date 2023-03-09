from typing import Any, Iterable, Iterator, Union


def iter_structure(
        structure: Union[Any, Iterable], *,
        item_types: Iterable[type] = ()
) -> Iterator[Any]:
    sub_items = None

    if structure not in item_types:
        try:
            sub_items = iter(structure)
        except TypeError:
            sub_items = None

    if sub_items is not None:
        for item in sub_items:
            yield from iter_structure(item, item_types=item_types)
    else:
        yield structure


def iter_zip_structures(
        *structures: Union[Any, Iterable],
        item_types: Iterable[type] = ()
) -> Iterator[tuple]:
    sub_items = None

    if structures[0] not in item_types:
        try:
            sub_items = zip(*[iter(item) for item in structures])
        except TypeError:
            sub_items = None

    if sub_items is not None:
        for item in sub_items:
            yield from iter_zip_structures(*item, item_types=item_types)
    else:
        yield structures


def restore_structure(
        values: Iterable[Any],
        structure: Union[Any, Iterable], *,
        item_types: Iterable[type] = ()
) -> Union[Any, tuple]:
    values = iter(values)

    sub_structures = None
    if structure not in item_types:
        try:
            sub_structures = iter(structure)
        except TypeError:
            sub_structures = None

    if sub_structures is not None:
        return tuple(restore_structure(values, sub_structure) for sub_structure in sub_structures)
    else:
        return next(values)
