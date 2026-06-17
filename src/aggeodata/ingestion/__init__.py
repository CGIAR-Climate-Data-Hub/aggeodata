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
]
