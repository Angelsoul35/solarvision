"""The defect taxonomy: the classes SolarVision can assign to a panel image.

Keeping the taxonomy in one place (rather than scattering string literals) means
the renderer, the feature layer, the detector and the API all agree on exactly
four mutually exclusive labels, and each ships with a plain-language description a
reviewer can read.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DefectClass(StrEnum):
    """The four classes. ``HEALTHY`` is the negative class; the rest are defects.

    These mirror the visual signatures seen in real electroluminescence (EL) and
    thermal inspection of photovoltaic modules, rendered here synthetically.
    """

    HEALTHY = "healthy"
    HOTSPOT = "hotspot"
    CRACK = "crack"
    DEAD_CELL = "dead_cell"

    @property
    def is_defect(self) -> bool:
        return self is not DefectClass.HEALTHY


@dataclass(frozen=True)
class DefectInfo:
    label: DefectClass
    title: str
    signature: str
    note: str


_CATALOG: dict[DefectClass, DefectInfo] = {
    DefectClass.HEALTHY: DefectInfo(
        label=DefectClass.HEALTHY,
        title="Healthy panel",
        signature="uniform cell grid, no localised bright or dark anomaly",
        note="the negative class; natural cell-to-cell variation and sensor noise only",
    ),
    DefectClass.HOTSPOT: DefectInfo(
        label=DefectClass.HOTSPOT,
        title="Hot spot",
        signature="a localised bright blob far above the surrounding cells",
        note="a shaded or shunted cell dissipates power and heats up, reading bright",
    ),
    DefectClass.CRACK: DefectInfo(
        label=DefectClass.CRACK,
        title="Micro-crack",
        signature="a thin, elongated dark streak crossing the cell grid",
        note="mechanical/thermal stress fractures the wafer; the dead side goes dark",
    ),
    DefectClass.DEAD_CELL: DefectInfo(
        label=DefectClass.DEAD_CELL,
        title="Dead cell",
        signature="a compact, solid dark block roughly the size of one cell",
        note="an inactive cell produces no signal and reads as a uniform dark region",
    ),
}


def defect_catalog() -> dict[DefectClass, DefectInfo]:
    """The full taxonomy, keyed by class."""
    return dict(_CATALOG)


def describe(label: DefectClass | str) -> DefectInfo:
    """Look up the metadata for a class, accepting the enum or its string value."""
    return _CATALOG[DefectClass(label)]


ALL_CLASSES: list[str] = [c.value for c in DefectClass]
DEFECT_CLASSES: list[str] = [c.value for c in DefectClass if c.is_defect]
