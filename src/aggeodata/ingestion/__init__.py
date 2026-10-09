"""aggeodata.ingestion — Data download layer (download-only, no cube-building)."""

from .chirps import CHIRPSDownloader
from .chirts import CHIRTSDownloader
from .agera5 import AgEra5Downloader
from .nasa_power import NASAPowerDownloader, NASAPowerS3Downloader
from .gee import GEEDownloader
from .climate import WeatherDownloadOrchestrator
from .soil import SoilGridsDownloader
from .boundaries import get_admin_boundary, list_admin_units
from .cf_names import CF_VARIABLE_MAP, AGERA5_CF_MAP, to_cf_name, rename_cf_vars
from .zarr_backend import (
    AGERA5_ARCO_GEO_URL,
    AGERA5_ARCO_TIME_URL,
    HF_CHIRPS_URL,
    HF_CHIRTS_URL,
    cds_storage_options,
    download_daily_from_zarr,
    open_remote_zarr,
    read_cds_api_key,
    subset_zarr,
)

__all__ = [
    # climate — product downloaders
    "CHIRPSDownloader",
    "CHIRTSDownloader",
    "AgEra5Downloader",
    "NASAPowerDownloader",
    "NASAPowerS3Downloader",
    "GEEDownloader",
    # climate — orchestrator
    "WeatherDownloadOrchestrator",
    # soil
    "SoilGridsDownloader",
    # boundaries
    "get_admin_boundary",
    "list_admin_units",
    # CF name utilities
    "CF_VARIABLE_MAP",
    "AGERA5_CF_MAP",
    "to_cf_name",
    "rename_cf_vars",
    # Analysis-ready Zarr backend (CHIRPS / CHIRTS / AgERA5)
    "HF_CHIRPS_URL",
    "HF_CHIRTS_URL",
    "AGERA5_ARCO_GEO_URL",
    "AGERA5_ARCO_TIME_URL",
    "open_remote_zarr",
    "subset_zarr",
    "download_daily_from_zarr",
    "read_cds_api_key",
    "cds_storage_options",
]
