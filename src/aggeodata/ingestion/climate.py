"""
aggeodata.ingestion.climate
============================

Climate ingestion orchestrator.

This module is the single entry point for multi-variable climate downloads.
Each data source has its own dedicated module; this module wires them together
via :class:`WeatherDownloadOrchestrator`.

Product modules:
    chirps     → :class:`~aggeodata.ingestion.chirps.CHIRPSDownloader`
    chirts     → :class:`~aggeodata.ingestion.chirts.CHIRTSDownloader`
    agera5     → :class:`~aggeodata.ingestion.agera5.AgEra5Downloader`
    nasa_power → :class:`~aggeodata.ingestion.nasa_power.NASAPowerS3Downloader`
                 :class:`~aggeodata.ingestion.nasa_power.NASAPowerDownloader`
"""

from __future__ import annotations

import logging
from pathlib import Path

from shapely.geometry import Polygon

from .agera5 import AgEra5Downloader, AGERA5_VARIABLE_MAP
from .chirps import CHIRPSDownloader
from .chirts import CHIRTSDownloader
from .nasa_power import NASAPowerDownloader, NASAPowerS3Downloader
from .gis_functions import from_polygon_2bbox

logger = logging.getLogger(__name__)

__all__ = [
    "WeatherDownloadOrchestrator",
    # re-export product downloaders for convenience
    "CHIRPSDownloader",
    "CHIRTSDownloader",
    "AgEra5Downloader",
    "NASAPowerDownloader",
    "NASAPowerS3Downloader",
]


class WeatherDownloadOrchestrator:
    """Orchestrate multi-variable climate data downloads.

    Routes each variable to the correct product downloader:

    * ``precipitation``   → :class:`~aggeodata.ingestion.chirps.CHIRPSDownloader`
    * ``tmax`` / ``tmin`` → :class:`~aggeodata.ingestion.chirts.CHIRTSDownloader`
    * AgERA5 variables    → :class:`~aggeodata.ingestion.agera5.AgEra5Downloader`
    * ``nasa_power``      → :class:`~aggeodata.ingestion.nasa_power.NASAPowerS3Downloader`

    Parameters
    ----------
    starting_date : str
        Start date ``"YYYY-MM-DD"``.
    ending_date : str
        End date ``"YYYY-MM-DD"``.
    xyxy : list[float] | None
        Bounding box ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
    output_folder : str | None
        Root folder for all downloads.
    aoi : Polygon | None
        Alternative to ``xyxy`` — bounding box derived from this polygon.

    Examples
    --------
    >>> orch = WeatherDownloadOrchestrator(
    ...     starting_date="2010-01-01",
    ...     ending_date="2019-12-31",
    ...     xyxy=[-90.5, 13.0, -88.5, 15.5],
    ...     output_folder="data/raw",
    ... )
    >>> orch.download({"precipitation": {}, "solar_radiation": {"source": "agera5"}})
    """

    def __init__(
        self,
        starting_date: str,
        ending_date: str,
        xyxy: list[float] | None = None,
        output_folder: str | None = None,
        aoi: Polygon | None = None,
    ) -> None:
        self.starting_date = starting_date
        self.ending_date = ending_date
        self.extent: list[float] = from_polygon_2bbox(aoi) if aoi else (xyxy or [])
        self.output_folder = output_folder
        if output_folder:
            Path(output_folder).mkdir(parents=True, exist_ok=True)

    def download(
        self,
        weather_variables: dict[str, dict],
        suffix: str | None = None,
        export_as_netcdf: bool = False,
        ncores: int = 4,
        agera5_version: str = "2_0",
        nasa_power_backend: str = "s3",
    ) -> dict[str, dict[str, str]]:
        """Download all requested climate variables.

        Parameters
        ----------
        weather_variables : dict[str, dict]
            Keys are variable names; values are option dicts.
            Recognised keys in the option dict:

            * ``source`` — ``"chirps"``, ``"chirts"``, ``"agera5"``, ``"nasa_power"``.
              Inferred automatically when omitted.
            * ``chirts_source`` — ``"era5"`` or ``"chirts"`` for CHIRTS downloads.
        suffix : str | None
            Optional folder name suffix (e.g. region code).
        export_as_netcdf : bool
            Stack AgERA5 zip files into per-year ``.nc`` after download.
        ncores : int
            Parallel workers per downloader.
        agera5_version : str
            AgERA5 product version (``"2_0"`` or ``"1_1"``).
        nasa_power_backend : str
            ``"s3"`` (default, Zarr) or ``"api"`` (REST tiles).

        Returns
        -------
        dict[str, dict[str, str]]
            ``{variable: {year_or_key: path}}``
        """
        results: dict[str, dict[str, str]] = {}

        for var_key, opts in weather_variables.items():
            out_folder = self._make_output_folder(var_key, suffix)
            source = opts.get("source") or self._infer_source(var_key)

            if source == "chirps":
                dl = CHIRPSDownloader()
                paths = dl.download(
                    extent=self.extent,
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    output_folder=out_folder,
                    ncores=ncores,
                )
                results[var_key] = paths

            elif source == "chirts":
                chirts_var = opts.get("variable", var_key.replace("temperature_", ""))
                dl_chirts = CHIRTSDownloader(
                    variables=[chirts_var] if chirts_var in {"tmax", "tmin"} else None,
                    source=opts.get("chirts_source", "era5"),
                )
                paths_t = dl_chirts.download(
                    extent=self.extent,
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    output_folder=out_folder,
                    ncores=ncores,
                )
                results[var_key] = {yr: p for var_paths in paths_t.values() for yr, p in var_paths.items()}

            elif source == "agera5":
                matched = next((k for k in AGERA5_VARIABLE_MAP if k in var_key), None)
                if matched is None:
                    logger.warning("AgERA5: unknown variable '%s', skipping.", var_key)
                    results[var_key] = {}
                    continue
                spec = AGERA5_VARIABLE_MAP[matched]
                dl_agera5 = AgEra5Downloader(version=agera5_version)
                paths = dl_agera5.download(
                    variable=spec["variable"],
                    statistic=spec.get("statistic"),
                    time=spec.get("time"),
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    output_folder=out_folder,
                    aoi_extent=self.extent,
                    ncores=ncores,
                )
                results[var_key] = paths
                if export_as_netcdf and paths:
                    years = sorted(int(y) for y in paths)
                    AgEra5Downloader.stack_annual_to_netcdf(out_folder, years[0], years[-1], out_folder)

            elif source == "nasa_power":
                params = opts.get("parameters") or [var_key]
                if nasa_power_backend == "s3":
                    dl_power: NASAPowerS3Downloader | NASAPowerDownloader = NASAPowerS3Downloader(parameters=params)
                else:
                    dl_power = NASAPowerDownloader(parameters=params)
                nc_path = dl_power.download(
                    extent=self.extent,
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    output_folder=out_folder,
                )
                results[var_key] = {"all": nc_path}

            else:
                logger.warning("Unknown source '%s' for variable '%s', skipping.", source, var_key)
                results[var_key] = {}

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_source(var_key: str) -> str:
        """Infer the data source from the variable name."""
        if "precipitation" in var_key:
            return "chirps"
        if var_key in {"tmax", "tmin"} or "temperature_t" in var_key:
            return "chirts"
        if var_key in AGERA5_VARIABLE_MAP or any(k in var_key for k in AGERA5_VARIABLE_MAP):
            return "agera5"
        return "nasa_power"

    def _make_output_folder(self, variable: str, suffix: str | None) -> str:
        name = f"{variable}_{suffix}_raw" if suffix else f"{variable}_raw"
        path = Path(self.output_folder or ".") / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
