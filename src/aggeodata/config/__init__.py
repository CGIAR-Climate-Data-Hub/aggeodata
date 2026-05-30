"""aggeodata.config — YAML-driven configuration layer."""

from .schemas import (
    AgeodataConfig,
    DatesConfig,
    SpatialConfig,
    VariableConfig,
    IngestionClimateConfig,
    ClimateConfig,
    SoilConfig,
    GeneralConfig,
    PathsConfig,
    PipelineConfig,
    GeneralInfoConfig,
    DataSummarizationConfig,
)
from .loader import load_config

__all__ = [
    "AgeodataConfig",
    "DatesConfig",
    "SpatialConfig",
    "VariableConfig",
    "IngestionClimateConfig",
    "ClimateConfig",
    "SoilConfig",
    "GeneralConfig",
    "PathsConfig",
    "PipelineConfig",
    "GeneralInfoConfig",
    "DataSummarizationConfig",
    "load_config",
]
