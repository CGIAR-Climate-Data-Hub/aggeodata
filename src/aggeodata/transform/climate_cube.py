"""
aggeodata.transform.climate_cube
==================================

Datacube Construction Layer — Climate.

Builds multi-temporal xarray Datasets from downloaded climate files.
No downloads happen here.

Key exports:

* :func:`stack_datacube_temporally`   — concatenate per-date Datasets along time
* :func:`set_climate_encoding`        — zlib compression encoding dict
* :class:`ClimateDataCube`            — multi-year, multi-variable climate cube builder
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Migrated from: ag_cube_cm.transform.weather_cube (MLTWeatherDataCube)
# TODO: full migration pending
# ---------------------------------------------------------------------------


def stack_datacube_temporally(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("stack_datacube_temporally: migration pending")


def set_climate_encoding(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("set_climate_encoding: migration pending")


class ClimateDataCube:
    """Multi-year, multi-variable climate datacube builder. Migration pending."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError("ClimateDataCube: migration pending")

__all__ = [
    "ClimateDataCube",
    "stack_datacube_temporally",
    "set_climate_encoding",
]
