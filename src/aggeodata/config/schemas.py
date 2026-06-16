"""
aggeodata.config.schemas
========================

Pydantic v2 models for aggeodata YAML configuration files.
"""

from __future__ import annotations

from typing import Annotated, Literal, List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Supported sources per CF variable
# ---------------------------------------------------------------------------

_VALID_SOURCES: frozenset[str] = frozenset(
    {"chirps", "chirts", "agera5", "nasa_power"}
)

# CF variables that each source can provide
_SOURCE_VARIABLES: dict[str, frozenset[str]] = {
    "chirps":     frozenset({"pr"}),
    "chirts":     frozenset({"tasmax", "tasmin"}),
    "agera5":     frozenset({
        "pr", "tasmax", "tasmin", "rsds", "hurs", "sfcWind", "tdps", "etr", "vp", "vpd",
        "hurs_06", "hurs_09", "hurs_12", "hurs_15", "hurs_18",
    }),
    "nasa_power": frozenset({"pr", "tasmax", "tasmin", "tas", "rsds", "hurs", "sfcWind"}),
}


# ---------------------------------------------------------------------------
# Per-variable config
# ---------------------------------------------------------------------------

class VariableConfig(BaseModel):
    """Configuration for a single CF climate variable."""

    source: Annotated[
        str,
        Field(description="Data provider: chirps | chirts | agera5 | nasa_power"),
    ]
    chirts_source: Annotated[
        Literal["era5", "chirts"],
        Field(default="era5", description="CHIRTS variant — only used when source=chirts"),
    ] = "era5"
    agera5_key: Annotated[
        str | None,
        Field(default=None, description="Override default CF→AgERA5 variable key"),
    ] = None
    nasa_power_param: Annotated[
        str | None,
        Field(default=None, description="Override default CF→NASA POWER parameter code"),
    ] = None

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in _VALID_SOURCES:
            raise ValueError(
                f"source '{v}' is not supported. "
                f"Valid sources: {sorted(_VALID_SOURCES)}"
            )
        return v


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class DatesConfig(BaseModel):
    starting_date: Annotated[str, Field(description="Start date YYYY-MM-DD")]
    ending_date: Annotated[str, Field(description="End date YYYY-MM-DD")]

    @field_validator("starting_date", "ending_date", mode="before")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        from datetime import datetime
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @model_validator(mode="after")
    def end_after_start(self) -> DatesConfig:
        from datetime import datetime
        s = datetime.strptime(self.starting_date, "%Y-%m-%d")
        e = datetime.strptime(self.ending_date, "%Y-%m-%d")
        if e <= s:
            raise ValueError("ending_date must be after starting_date")
        return self


class SpatialConfig(BaseModel):
    spatial_file: Annotated[str | None, Field(default=None)]
    extent: Annotated[
        list[float] | None,
        Field(default=None, description="[xmin, ymin, xmax, ymax] EPSG:4326"),
    ] = None

    @model_validator(mode="after")
    def at_least_one(self) -> SpatialConfig:
        if not self.spatial_file and not self.extent:
            raise ValueError("Either spatial_file or extent must be provided")
        return self

    @field_validator("extent", mode="before")
    @classmethod
    def validate_extent(cls, v):
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("extent must have 4 values: [xmin, ymin, xmax, ymax]")
        xmin, ymin, xmax, ymax = v
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("extent: xmin < xmax and ymin < ymax required")
        return v


class IngestionClimateConfig(BaseModel):
    variables: Annotated[
        dict[str, VariableConfig],
        Field(description="CF variable name → source config"),
    ]

    @model_validator(mode="after")
    def validate_source_variable_pairs(self) -> IngestionClimateConfig:
        for cf_var, cfg in self.variables.items():
            allowed = _SOURCE_VARIABLES.get(cfg.source, frozenset())
            if allowed and cf_var not in allowed:
                raise ValueError(
                    f"Variable '{cf_var}' is not available from source '{cfg.source}'. "
                    f"Supported CF variables for {cfg.source}: {sorted(allowed)}"
                )
        return self


class SoilConfig(BaseModel):
    enabled: bool = False
    layers: Annotated[
        list[str],
        Field(default=["clay", "sand"], description="SoilGrids variable names"),
    ] = ["clay", "sand"]
    depths: Annotated[
        list[str],
        Field(default=["0-5"], description="Depth intervals, e.g. '0-5', '5-15'"),
    ] = ["0-5"]


class GeneralConfig(BaseModel):
    suffix: Annotated[str, Field(default="", description="Appended to output folder names")] = ""
    ncores: Annotated[int, Field(default=2, ge=1)] = 2
    task: Annotated[
        Literal["download", "datacube"],
        Field(default="download"),
    ] = "download"
    reference_variable: Annotated[
        str,
        Field(default="pr", description="CF variable used as spatial reference for the datacube"),
    ] = "pr"
    agera5_version: Annotated[
        Literal["1_1", "2_0"],
        Field(default="2_0"),
    ] = "2_0"
    nasa_power_backend: Annotated[
        Literal["s3", "rest"],
        Field(default="s3", description="s3=Zarr (faster), rest=tile API"),
    ] = "s3"
    target_crs: Annotated[
        str,
        Field(default="EPSG:4326", description="Output CRS for the datacube (any PROJ/EPSG string)"),
    ] = "EPSG:4326"
    target_resolution: Annotated[
        Optional[float],
        Field(default=None, description="Output spatial resolution in CRS units (degrees for EPSG:4326). None keeps native resolution."),
    ] = None


class PathsConfig(BaseModel):
    output_path: Annotated[str, Field(default="data/raw/")] = "data/raw/"

class AgeodataConfig(BaseModel):
    """Complete aggeodata YAML configuration."""

    DATES: DatesConfig
    SPATIAL_INFO: SpatialConfig
    CLIMATE: IngestionClimateConfig
    SOIL: SoilConfig = SoilConfig()
    GENERAL: GeneralConfig = GeneralConfig()
    PATHS: PathsConfig = PathsConfig()

    @model_validator(mode="after")
    def reference_variable_in_climate(self) -> AgeodataConfig:
        ref = self.GENERAL.reference_variable
        if ref and ref not in self.CLIMATE.variables:
            raise ValueError(
                f"reference_variable '{ref}' is not listed under CLIMATE.variables. "
                f"Available: {list(self.CLIMATE.variables)}"
            )
        return self

    def get_extent(self) -> list[float]:
        """Return [xmin, ymin, xmax, ymax] from spatial_file or extent."""
        if self.SPATIAL_INFO.spatial_file:
            import geopandas as gpd
            gdf = gpd.read_file(self.SPATIAL_INFO.spatial_file)
            b = gdf.total_bounds  # (xmin, ymin, xmax, ymax)
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        return list(self.SPATIAL_INFO.extent)

    def var_folder(self, cf_var: str) -> str:
        """Return the download output folder for a CF variable."""
        import os
        name = f"{cf_var}_{self.GENERAL.suffix}_raw" if self.GENERAL.suffix else f"{cf_var}_raw"
        return os.path.join(self.PATHS.output_path, name)


class ClimateIndex(BaseModel):
    """
    Climate index configuration (e.g., SPI, SPEI, ScPDSI).
    """
    name: str
    description: Optional[str] = None
    meteorological_variables: List[str]
    parameters: Optional[Dict[str, Any]] = None


class ClimateSummarisation(BaseModel):
    """
    Configuration for basic structural aggregation functions over climate layers.
    """
    description: Optional[str] = None
    meteorological_variable: str 
    summary_function: str  # e.g., "mean", "sum", "max", "min", "std"
    parameters: Optional[Dict[str, Any]] = None

    # Improvement 1: Protect summary_function names against typos at initialization
    @field_validator('summary_function')
    @classmethod
    def validate_summary_function(cls, v: str) -> str:
        allowed = {"mean", "sum", "max", "min", "std", "median", "var"}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(
                f"Unsupported summary_function '{v}'. Must be one of {allowed}"
            )
        return v_lower


class ClimateConfig(BaseModel):
    """
    Consolidated configuration for environmental input data engines.
    """
    climate_data_path: Annotated[Path, Field(description='Root path to the input climate dataset directory')]
    indices: List[ClimateIndex] = Field(default_factory=list)
    summarizations: List[ClimateSummarisation] = Field(default_factory=list)
    

class GeneralInfoConfig(BaseModel):
    """
    Global process execution parameters.
    """
    n_cores: Annotated[int, Field(description='Number of CPU cores to allocate for parallel worker pools')]
    output_path: Annotated[Path, Field(description='Directory path where final results will be serialized')]
    year_oi: Annotated[Optional[int], Field(default=None, description="Year of interest for filtering climate data layers (e.g., 2014). If None, no year-based filtering will be applied.")]
    
    spatial_data_path: Annotated[
        Optional[Path], 
        Field(default=None, description="Path to coordinate layers (e.g., shapefile/GeoJSON) for zonal/point extractions.")
    ]

    # Improvement 2: Prevent thread-pool collapse bugs
    @field_validator('n_cores')
    @classmethod
    def validate_cores(cls, v: int) -> int:
        if v < -1 or v == 0:
            raise ValueError("n_cores must be a positive integer, or -1 to utilize all available processors.")
        return v

class DataSummarizationConfig(BaseModel):
    """
    Configuration details handling temporal clipping and agricultural/experimental timeline syncs.
    """
    field_data_source: Annotated[
        Path, 
        Field(description="Source file mapping experimental observation sites to rows")
    ]
    column_starting_date: Annotated[Optional[str], Field(default=None)]
    column_ending_date: Annotated[Optional[str], Field(default=None)]
    temporal_window: Annotated[Optional[int], Field(default=6)]
    nmonths_lookahead: Annotated[Optional[int], Field(default=0)]
    nmonths_lookback: Annotated[Optional[int], Field(default=6)]
    output_filename: Annotated[Optional[str], Field(default="extracted_climate_data.csv", description="Filename for the extracted climate data CSV output")]
    
    @model_validator(mode='after')
    def validate_date_columns(self) -> 'DataSummarizationConfig':
        if self.field_data_source and not (self.column_starting_date or self.column_ending_date):
            raise ValueError(
                "When 'field_data_source' is provided, you must provide either "
                "'column_starting_date' or 'column_ending_date' to execute aggregations."
            )
        return self


class PipelineConfig(BaseModel):
    """
    Master declarative profile orchestrating the complete calculation graph.
    """
    general_info: GeneralInfoConfig
    climate_config: ClimateConfig
    data_summarization: DataSummarizationConfig
