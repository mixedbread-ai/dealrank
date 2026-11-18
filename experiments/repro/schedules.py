"""Schedule specs in experiment configs.

A spec is a mapping with a ``type`` and the schedule's keyword arguments, for
example ``{"type": "dealrank", "quota": 4}``. ``blitzrank_package`` builds
:class:`~repro.blitzrank_package.BlitzRankPackage`; every other type goes to
:func:`dealrank.build_schedule`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dealrank import Schedule
from dealrank import build_schedule as build_library_schedule
from repro.blitzrank_package import BlitzRankPackage

PACKAGE_TYPE = "blitzrank_package"


def build_schedule(spec: Mapping[str, Any]) -> Schedule:
    """Build the schedule a config spec describes."""
    spec = dict(spec)
    if spec.get("type") == PACKAGE_TYPE:
        spec.pop("type")
        return BlitzRankPackage(**spec)
    return build_library_schedule(spec)


def uses_package(spec: Mapping[str, Any]) -> bool:
    """Whether ``spec`` runs the released BlitzRank package."""
    return spec.get("type") == PACKAGE_TYPE


def package_available() -> bool:
    """Whether the released BlitzRank package can be imported."""
    try:
        import blitzrank  # noqa: F401
    except ImportError:
        return False
    return True
