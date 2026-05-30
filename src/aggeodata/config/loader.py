"""aggeodata.config.loader — YAML → AgeodataConfig | PipelineConfig."""

from __future__ import annotations

from pathlib import Path

from .schemas import AgeodataConfig, PipelineConfig


def load_config(path: str | Path) -> AgeodataConfig | PipelineConfig:
    """Load and validate an aggeodata YAML config file.

    Detects the config type from top-level keys:
    - ``climate_config`` present → :class:`PipelineConfig` (summarization workflow)
    - otherwise → :class:`AgeodataConfig` (ingestion/datacube workflow)

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    AgeodataConfig | PipelineConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pydantic.ValidationError
        If the YAML content fails validation.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "climate_config" in raw:
        return PipelineConfig.model_validate(raw)
    return AgeodataConfig.model_validate(raw)
