"""Shared value-copy helpers for restorable solver state."""


def snapshot_triplet(value, previous, second_previous, copy_value):
    """Copy distinct objects while preserving aliases between time levels."""
    current = copy_value(value)
    prior = current if previous is value else copy_value(previous)
    if second_previous is value:
        older = current
    elif second_previous is previous:
        older = prior
    else:
        older = copy_value(second_previous)
    return current, prior, older
