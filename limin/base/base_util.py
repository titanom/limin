from typing import TypeVar


T = TypeVar("T")


def get_first_element(list: list[T]) -> T | None:
    if len(list) == 0:
        return None
    return list[0]


def get_last_element(list: list[T]) -> T | None:
    if len(list) == 0:
        return None
    return list[-1]
