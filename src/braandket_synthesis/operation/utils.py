from typing import Any, Iterable, Iterator, Union


def iter_structured(items: Union[Any, Iterable], *, item_types: Iterable[type] = ()) -> Iterator[Any]:
    sub_items = None

    if items not in item_types:
        try:
            sub_items = iter(items)
        except TypeError:
            sub_items = None

    if sub_items is not None:
        for item in sub_items:
            yield from iter_structured(item, item_types=item_types)
    else:
        yield items


def iter_structured_zip(*items: Union[Any, Iterable], item_types: Iterable[type] = ()) -> Iterator[Any]:
    sub_items = None

    if items[0] not in item_types:
        try:
            sub_items = zip(*[iter(item) for item in items])
        except TypeError:
            sub_items = None

    if sub_items is not None:
        for item in sub_items:
            yield from iter_structured_zip(*item, item_types=item_types)
    else:
        yield items
