"""Helpers for enabling/disabling user Balance objects in solvers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_ignore_balances(ignore_balances: Any) -> frozenset[str] | str | None:
    """Normalize the public ``ignore_balances`` solver argument.

    Accepted values are:

    * ``None``: include every user ``Balance``.
    * ``"all"``: skip every user ``Balance``.
    * iterable of names: skip only balances whose ``name`` is listed.
    """
    if ignore_balances is None:
        return None

    if ignore_balances == "all":
        return "all"

    if isinstance(ignore_balances, str):
        raise ValueError(
            "ignore_balances must be None, 'all', or an iterable of balance names. "
            f"Got string {ignore_balances!r}."
        )

    if not isinstance(ignore_balances, Iterable):
        raise TypeError(
            "ignore_balances must be None, 'all', or an iterable of balance names. "
            f"Got {type(ignore_balances).__name__}."
        )

    names: set[str] = set()
    for name in ignore_balances:
        if not isinstance(name, str):
            raise TypeError(
                "ignore_balances entries must be balance-name strings. "
                f"Got {type(name).__name__}."
            )
        names.add(name)

    return frozenset(names)


def filter_user_balances(balance_list: Iterable[Any], ignore_balances: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return ``(active_balances, ignored_balances)`` for a balance list."""
    balances = tuple(balance_list)
    ignored = normalize_ignore_balances(ignore_balances)

    if ignored is None:
        return balances, ()

    if ignored == "all":
        return (), balances

    active_balances: list[Any] = []
    ignored_balances: list[Any] = []
    missing_names = set(ignored)

    for balance in balances:
        balance_name = getattr(balance, "name", None)
        if balance_name in ignored:
            ignored_balances.append(balance)
            missing_names.discard(balance_name)
        else:
            active_balances.append(balance)

    if missing_names:
        available_names = sorted(str(getattr(balance, "name", balance)) for balance in balances)
        missing_text = ", ".join(sorted(missing_names))
        available_text = ", ".join(available_names) if available_names else "<none>"
        raise ValueError(
            "ignore_balances listed balance names that are not in this network.\n"
            f"Missing: {missing_text}\n"
            f"Available: {available_text}"
        )

    return tuple(active_balances), tuple(ignored_balances)
