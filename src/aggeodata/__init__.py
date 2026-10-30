"""aggeodata — Multi-source climate and soil data ingestion and datacube transforms."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aggeodata")  # read from pyproject.toml at install time
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "unknown"

from aggeodata.ingestion import (
    CHIRPSDownloader,
    CHIRTSDownloader,
    AgEra5Downloader,
    NASAPowerDownloader,
    NASAPowerS3Downloader,
    SoilGridsDownloader,
    WeatherDownloadOrchestrator,
    get_admin_boundary,
    list_admin_units,
)
from aggeodata.transform import (
    ClimateDataCube,
    SoilDataCubeBuilder,
    stack_datacube_temporally,
)

__all__ = [
    # ingestion — climate (product downloaders)
    "CHIRPSDownloader",
    "CHIRTSDownloader",
    "AgEra5Downloader",
    "NASAPowerDownloader",
    "NASAPowerS3Downloader",
    # ingestion — climate (orchestrator)
    "WeatherDownloadOrchestrator",
    # ingestion — soil
    "SoilGridsDownloader",
    # ingestion — boundaries
    "get_admin_boundary",
    "list_admin_units",
    # transform
    "ClimateDataCube",
    "SoilDataCubeBuilder",
    "stack_datacube_temporally",
]
