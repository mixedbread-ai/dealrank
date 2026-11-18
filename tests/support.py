"""Helpers shared by the tests: passages whose text encodes their true rank."""

from __future__ import annotations


def doc(i: int) -> str:
    """Passage text that encodes its true rank; ``doc(0)`` is the best."""
    return f"passage {i}"


def true_rank(passage: str) -> int:
    return int(passage.split()[1])
