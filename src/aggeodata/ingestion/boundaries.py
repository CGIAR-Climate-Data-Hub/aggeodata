"""
aggeodata.ingestion.boundaries
================================

Admin boundary resolution via the GeoBoundaries API.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Migrated from: ag_cube_cm.ingestion.boundaries
# ---------------------------------------------------------------------------


__all__ = ["get_admin_boundary", "list_admin_units"]


import io
import json
import logging
from functools import lru_cache
from typing import Any

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

_GEOBOUNDARIES_BASE = "https://www.geoboundaries.org/api/current/gbOpen"
_REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str) -> Any:
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _geojson_url(country_code: str, adm_level: int) -> str:
    """Return the GeoJSON download URL from the GeoBoundaries metadata endpoint."""
    url = f"{_GEOBOUNDARIES_BASE}/{country_code.upper()}/ADM{adm_level}/"
    logger.info("GeoBoundaries: fetching metadata from %s", url)
    meta = _fetch_json(url)
    geojson_url = meta.get("gjDownloadURL", "")
    if not geojson_url:
        raise ValueError(
            f"GeoBoundaries returned no GeoJSON download URL for "
            f"{country_code} ADM{adm_level}. Response: {meta}"
        )
    return geojson_url


@lru_cache(maxsize=32)
def _fetch_names_cached(country_code: str, adm_level: int) -> list[dict[str, Any]]:
    """Download and cache only the feature properties (no geometry).

    Much faster than ``_fetch_geojson_cached`` for name lookups — avoids
    parsing polygon coordinates entirely.
    """
    geojson_url = _geojson_url(country_code, adm_level)
    logger.info("GeoBoundaries: downloading properties from %s", geojson_url)
    geojson_data = _fetch_json(geojson_url)
    return [f["properties"] for f in geojson_data.get("features", [])]


@lru_cache(maxsize=32)
def _fetch_geojson_cached(country_code: str, adm_level: int) -> gpd.GeoDataFrame:
    """Download and cache the full admin boundary GeoDataFrame.

    The ``@lru_cache`` means the network request fires only once per
    (country_code, adm_level) pair per Python session.  Subsequent calls
    for the same country/level are instant.
    """
    geojson_url = _geojson_url(country_code, adm_level)
    logger.info("GeoBoundaries: downloading GeoJSON from %s", geojson_url)
    geojson_data = _fetch_json(geojson_url)
    gdf = gpd.read_file(io.StringIO(json.dumps(geojson_data)))
    logger.info(
        "GeoBoundaries: loaded %d features  columns=%s",
        len(gdf), list(gdf.columns),
    )
    return gdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_admin_units(
    country_code: str,
    adm_level: int = 2,
    name_column: str | None = None,
) -> list[str]:
    """Return the names of all administrative units at a given level.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. ``'MWI'``).
    adm_level : int
        Administrative level to query.  Typical values:
        1 = region/province, 2 = district/department, 3 = sub-district.
        Default: 2.
    name_column : str | None
        Column in the GeoDataFrame that holds unit names.  When ``None``,
        the function tries ``'shapeName'``, ``'ADM2_EN'``, ``'NAME_2'`` in
        that order.

    Returns
    -------
    list[str]
        Sorted list of unit names.
    """
    props = _fetch_names_cached(country_code, adm_level)
    if not props:
        return []
    col = name_column or _detect_name_column_from_props(props[0])
    return sorted({p[col] for p in props if p.get(col) is not None})


def get_admin_boundary(
    country_code: str,
    feature_name: str,
    adm_level: int = 2,
    name_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Return the boundary polygon for a single administrative unit.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. ``'MWI'``).
    feature_name : str
        Name of the unit to extract (case-insensitive prefix match).
        Must match exactly one feature after matching; raises ``ValueError``
        if zero or multiple features match.
    adm_level : int
        Administrative level.  Default: 2 (district/department/county).
    name_column : str | None
        Column to match against.  Auto-detected when ``None``.

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame in EPSG:4326, ready to pass to
        :func:`~ag_cube_cm.spatial.raster_ops.get_roi_data`.

    Raises
    ------
    ValueError
        If no feature matches, or more than one feature matches and the
        match is ambiguous.
    """
    gdf = _fetch_geojson_cached(country_code, adm_level)
    col = name_column or _detect_name_column(gdf)

    # Case-insensitive exact match first
    mask_exact = gdf[col].str.lower() == feature_name.lower()
    matches = gdf[mask_exact]

    if len(matches) == 0:
        # Fall back to prefix match so "Zomba" matches "Zomba District"
        mask_prefix = gdf[col].str.lower().str.startswith(feature_name.lower())
        matches = gdf[mask_prefix]

    if len(matches) == 0:
        available = sorted(gdf[col].dropna().tolist())
        raise ValueError(
            f"No ADM{adm_level} unit matching '{feature_name}' found for "
            f"{country_code}.  Available units:\n  {available}"
        )

    if len(matches) > 1:
        names = matches[col].tolist()
        raise ValueError(
            f"'{feature_name}' matched {len(matches)} features for "
            f"{country_code} ADM{adm_level}: {names}.  "
            f"Provide a more specific name."
        )

    result = matches.copy()
    if result.crs is None or str(result.crs).upper() != "EPSG:4326":
        result = result.to_crs("EPSG:4326")

    logger.info(
        "Admin boundary: %s / %s  bbox=%s",
        country_code, feature_name,
        [round(v, 4) for v in result.total_bounds.tolist()],
    )
    return result.reset_index(drop=True)


def _detect_name_column_from_props(props: dict[str, Any]) -> str:
    """Detect the name column from a single feature's properties dict."""
    candidates = ["shapeName", "ADM2_EN", "ADM1_EN", "NAME_2", "NAME_1",
                  "name", "NAME", "admin2Name", "admin1Name"]
    for col in candidates:
        if col in props:
            return col
    # Last resort: first string-valued key
    for key, val in props.items():
        if isinstance(val, str):
            return key
    raise ValueError(
        f"Cannot detect a name column in feature properties. Keys: {list(props.keys())}"
    )


def _detect_name_column(gdf: gpd.GeoDataFrame) -> str:
    """Detect which column holds the unit name."""
    if len(gdf) > 0:
        try:
            return _detect_name_column_from_props(
                {c: gdf[c].iloc[0] for c in gdf.columns if c != "geometry"}
            )
        except ValueError:
            pass
    # Fallback: first non-geometry object column
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == object:
            return col
    raise ValueError(
        f"Cannot detect a name column in GeoDataFrame. "
        f"Columns: {list(gdf.columns)}"
    )
