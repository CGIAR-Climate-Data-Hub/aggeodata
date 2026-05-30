"""
aggeodata.transform.soil_cube
================================

Datacube Construction Layer — Soil.

Builds multi-depth xarray Datasets from downloaded SoilGrids files.
No downloads happen here.

Key exports:

* :data:`TEXTURE_CLASSES`                     — USDA texture class lookup
* :func:`find_soil_textural_class_in_nparray` — vectorized USDA classification
* :func:`calculate_rgf`                        — Root Growth Factor
* :func:`create_depth_dimension`               — stack layers along depth dim
* :func:`get_layer_texture`                    — add USDA textural class layer
* :class:`SoilDataCubeBuilder`                 — multi-depth soil cube builder
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Migrated from: ag_cube_cm.transform.soil_cube
# TODO: full migration pending
# ---------------------------------------------------------------------------

TEXTURE_CLASSES: dict = {}


def create_depth_dimension(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("create_depth_dimension: migration pending")


def calculate_rgf(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("calculate_rgf: migration pending")


def find_soil_textural_class_in_nparray(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("find_soil_textural_class_in_nparray: migration pending")


def get_layer_texture(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("get_layer_texture: migration pending")


class SoilDataCubeBuilder:
    """Multi-depth soil datacube builder. Migration pending."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError("SoilDataCubeBuilder: migration pending")

__all__ = [
    "SoilDataCubeBuilder",
    "create_depth_dimension",
    "calculate_rgf",
    "find_soil_textural_class_in_nparray",
    "get_layer_texture",
    "TEXTURE_CLASSES",
]
