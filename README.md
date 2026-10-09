# aggeodata

Multi-source climate and soil data ingestion and datacube builder for agricultural and climate research.

Downloads daily gridded data from **CHIRPS**, **CHIRTS**, **AgERA5**, **NASA POWER**, and **SoilGrids**,
then assembles them into analysis-ready NetCDF datacubes aligned to a common grid and CRS.

---

## Data sources

| Source | Variables | Resolution | Period | API key? |
|--------|-----------|------------|--------|----------|
| [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | Precipitation (`pr`) | 0.05 deg | 1981–present | No |
| [CHIRTS](https://www.chc.ucsb.edu/data/chirts) | Tmax (`tasmax`), Tmin (`tasmin`) | 0.05 deg | 1983–present | No |
| [AgERA5](https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators) | Solar radiation (`rsds`), wind, RH, VPD | 0.1 deg | 1979–present | Yes — CDS |
| [NASA POWER](https://power.larc.nasa.gov/) | Solar radiation (`rsds`), wind, Tmax, Tmin | 0.5 deg | 1981–present | No |
| [SoilGrids](https://soilgrids.org/) | clay, sand, silt, bdod, cfvo, soc, phh2o, wv0010, wv0033, wv1500 | 250 m (~0.002 deg) | Static | No |

### Faster downloads via analysis-ready Zarr datacubes

The official endpoints hand out one file per day (CHIRPS/CHIRTS, throttled to a single
connection) or queue a request per year (AgERA5 via CDS), so a multi-year request takes
hours. The same archives are published as consolidated **Zarr datacubes**, where a
bounding box plus a date range is one indexed read:

| Source | Store | Variables | Credentials |
|--------|-------|-----------|-------------|
| CHIRPS v3 daily | [`aaguilar90/chirps-v3-daily-rnl`](https://huggingface.co/datasets/aaguilar90/chirps-v3-daily-rnl) | `precip` | none |
| CHIRTS-ERA5 daily | [`aaguilar90/chirts-era5-daily`](https://huggingface.co/datasets/aaguilar90/chirts-era5-daily) | `Tmax`, `Tmin` | none |
| AgERA5 | [ECMWF ARCO datastore](https://arco.datastores.ecmwf.int/) | all 16 variables aggeodata exposes | same `~/.cdsapirc` key |

Turn it on with one line in the config:

```yaml
GENERAL:
  use_zarr: true    # applies to every chirps / chirts / agera5 variable
```

or per variable:

```yaml
CLIMATE:
  variables:
    pr:
      source: chirps
      use_zarr: true
    tasmax:
      source: chirts
      use_zarr: false       # this one stays on the UCSB servers
    rsds:
      source: agera5
      use_zarr: true
      agera5_chunking: geo  # "geo" (default) | "time" — see below
```

Every backend writes the **same per-day NetCDF layout** in the same native units, so
`run_datacube` is unchanged. Requires the `download` extra (`zarr`).

`use_huggingface` / `hf_url` are still accepted as aliases for `use_zarr` / `zarr_url`.

**AgERA5 notes**

* The ARCO store publishes 28 variables; aggeodata maps the 16 it already exposes
  through `AGERA5_VARIABLE_MAP`. To add one of the others (cloud cover, snow thickness,
  precipitation duration fractions, day/night temperature means), add an entry to both
  `AGERA5_VARIABLE_MAP` and `AGERA5_ARCO_VARIABLES`.
* `agera5_chunking` picks the store: `geo` is chunked `(2112 time, 16 lat, 32 lon)` and
  suits many days over a modest AOI — the usual request, and the default. `time` is
  chunked `(1, 1024, 1024)` and only wins for a few days over a very large AOI.
* Values keep AgERA5's native units, same as the CDS backend (temperatures in **K**,
  radiation in J m-2 d-1).

**CHIRPS / CHIRTS notes**

* The mirrors carry CHIRPS v3 daily and CHIRTS-**ERA5** daily only. `chirts_source: chirts`
  (the original CHIRTS-daily v1.0) and `frequency="monthly"` are rejected with the
  Zarr backend — use the per-day one for those.
* Temperatures in the mirror are stored rounded to 0.1 °C; the UCSB GeoTIFFs carry
  full float32 precision (max difference 0.05 °C).
* For precipitation the two backends serve **different products**: the mirror is
  CHIRPS v3 (station-blended), while this repo's UCSB v3 endpoint is
  [CHIRP v3](https://data.chc.ucsb.edu/products/CHIRP-v3.0/) (satellite-only, no
  station blending). Pick one and keep it for the whole time series.

**All sources** — don't mix backends inside one variable folder. Clipping with rasterio
or a server-side `area` request can keep one extra edge pixel that the Zarr coordinate
slice drops, so the per-day grids differ. The downloader warns if it spots this.

---

## Install

Install directly from GitHub:

```bash
# Core + download extras (required for data download)
pip install "aggeodata[download,mcp] @ git+https://github.com/CGIAR-Climate-Data-Hub/aggeodata.git"
```

Optional extras:

```bash
# Climate indices (xclim)
pip install "aggeodata[download,indices] @ git+https://github.com/CGIAR-Climate-Data-Hub/aggeodata.git"

# Everything
pip install "aggeodata[all] @ git+https://github.com/CGIAR-Climate-Data-Hub/aggeodata.git"
```

Requires Python >= 3.10.

---

## AgERA5 setup (one-time)

AgERA5 requires a free CDS account. Register at
[cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/), then create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <YOUR-UID>:<YOUR-API-KEY>
```

The same key authenticates the ECMWF ARCO Zarr store used by `use_zarr: true`, so no
extra setup is needed for the faster AgERA5 backend.

All other sources (CHIRPS, CHIRTS, NASA POWER, SoilGrids) require no registration.

---

## Quick start

### Download climate data (pipeline)

The easiest way is through the pipeline, driven by a YAML config:

```python
from aggeodata.pipelines.download import run_download
from aggeodata.pipelines.datacube import run_datacube

# 1. Write a config (or use a dict)
import yaml, pathlib

config = {
    "DATES": {"starting_date": "2020-01-01", "ending_date": "2022-12-31"},
    "SPATIAL_INFO": {"extent": [-87.5, 14.2, -87.2, 14.5]},  # [xmin, ymin, xmax, ymax]
    "CLIMATE": {
        "variables": {
            "pr":     {"source": "chirps"},
            "tasmax": {"source": "chirts"},
            "tasmin": {"source": "chirts"},
            "rsds":   {"source": "nasa_power"},  # or "agera5" if CDS key configured
        }
    },
    "GENERAL": {
        "suffix": "hnd_small",
        "ncores": 2,
        "task": "download",
        "reference_variable": "pr",
        "target_crs": "EPSG:4326",
        "use_zarr": True,  # CHIRPS/CHIRTS/AgERA5 from Zarr datacubes (much faster)
    },
    "PATHS": {"output_path": "D:/data/hnd_small/climate_raw"},
}

cfg_path = "D:/data/hnd_small/config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)

# 2. Download raw files
run_download(cfg_path)

# 3. Build aligned NetCDF datacube
nc_path = run_datacube(cfg_path)
print("Weather cube:", nc_path)
# -> D:/data/hnd_small/climate_raw/climate_hnd_small_2020_2022.nc
```

### Download soil data (SoilGrids)

```python
from aggeodata.ingestion.soil import SoilGridsDownloader
from aggeodata.transform.soil_cube import SoilDataCubeBuilder

bbox = [-87.5, 14.2, -87.2, 14.5]  # [xmin, ymin, xmax, ymax]

# 1. Download GeoTIFF files from SoilGrids REST API
dl = SoilGridsDownloader(
    soil_layers=["clay", "sand", "silt", "bdod", "cfvo",
                 "soc", "phh2o", "wv0010", "wv0033", "wv1500"],
    depths=["0-5", "5-15", "15-30", "30-60", "60-100"],
    output_folder="D:/data/hnd_small/soil_raw",
)
dl.download(boundaries=bbox)

# 2. Stack into a single NetCDF datacube
builder = SoilDataCubeBuilder(
    data_folder="D:/data/hnd_small/soil_raw",
    variables=["clay", "sand", "silt", "bdod", "cfvo",
               "soc", "phh2o", "wv0010", "wv0033", "wv1500"],
    reference_variable="wv1500",
    target_crs="EPSG:4326",
)
builder.build_and_save(
    output_path="D:/data/hnd_small",
    filename="soil_hnd_small.nc",
)
# -> D:/data/hnd_small/soil_hnd_small.nc
```

### Download individual sources directly

**CHIRPS (precipitation)**
```python
from aggeodata.ingestion.chirps import CHIRPSDownloader

dl = CHIRPSDownloader(backend="hf")   # "ucsb" (default) for the official server
dl.download(
    extent=[-87.5, 14.2, -87.2, 14.5],
    starting_date="2020-01-01",
    ending_date="2022-12-31",
    output_folder="D:/data/chirps_raw",
    ncores=4,                         # HF backend: concurrent time-batches
)
```

**CHIRTS (temperature)**
```python
from aggeodata.ingestion.chirts import CHIRTSDownloader

dl = CHIRTSDownloader(variables=["tmax", "tmin"], backend="hf")
dl.download(extent=[-87.5, 14.2, -87.2, 14.5],
            starting_date="2020-01-01", ending_date="2022-12-31",
            output_folder="D:/data/chirts_raw", ncores=4)
```

**AgERA5 (requires CDS key — ARCO Zarr backend)**
```python
from aggeodata.ingestion.agera5 import AGERA5_VARIABLE_MAP, AgEra5Downloader

spec = AGERA5_VARIABLE_MAP["solar_radiation"]
dl = AgEra5Downloader(backend="zarr", chunking="geo")
dl.download(
    variable=spec["variable"], statistic=spec.get("statistic"), time=spec.get("time"),
    starting_date="2020-01-01", ending_date="2022-12-31",
    output_folder="D:/data/agera5_raw", aoi_extent=[-87.5, 14.2, -87.2, 14.5],
    ncores=4, config_key="solar_radiation",
)
```

**Read a Zarr store directly** (no files written)
```python
from aggeodata.ingestion import (
    AGERA5_ARCO_GEO_URL, HF_CHIRTS_URL,
    cds_storage_options, open_remote_zarr, subset_zarr,
)

# Public mirror — no credentials
ds = open_remote_zarr(HF_CHIRTS_URL)
sub = subset_zarr(ds, ["Tmax"], extent=[33.9, -4.7, 41.9, 5.5],
                  starting_date="2024-03-01", ending_date="2024-05-31").load()

# AgERA5 — bearer token from ~/.cdsapirc, latitude/longitude coordinate names
ds = open_remote_zarr(AGERA5_ARCO_GEO_URL, storage_options=cds_storage_options())
sub = subset_zarr(ds, ["Solar_Radiation_Flux"], extent=[33.9, -4.7, 41.9, 5.5],
                  starting_date="2024-03-01", ending_date="2024-05-31",
                  lat_name="latitude", lon_name="longitude").load()
```

**NASA POWER (solar radiation, no API key)**
```python
from aggeodata.ingestion.nasa_power import NASAPowerDownloader

dl = NASAPowerDownloader(parameters=["ALLSKY_SFC_SW_DWN"])
dl.download(
    extent=[-87.5, 14.2, -87.2, 14.5],
    starting_date="2020-01-01",
    ending_date="2022-12-31",
    output_folder="D:/data/nasa_power_raw",
)
```

**AgERA5 (requires CDS key)**
```python
from aggeodata.ingestion.agera5 import AgERA5Downloader

dl = AgERA5Downloader(variable="rsds", output_folder="D:/data/agera5_raw")
dl.download(extent=[-87.5, 14.2, -87.2, 14.5],
            starting_date="2020-01-01", ending_date="2022-12-31")
```

---

## Output datacube structure

All pipelines produce NetCDF files readable with xarray:

```python
import xarray as xr

# Weather cube
wds = xr.open_dataset("climate_hnd_small_2020_2022.nc")
# dims: time x y | vars: pr, tasmax, tasmin, rsds

# Soil cube
sds = xr.open_dataset("soil_hnd_small.nc")
# dims: depth x y | vars: clay, sand, silt, bdod, ...
```

### Exporting the climate cube as Zarr

`run_datacube` can also write a **Zarr v3 store** laid out like the
[GeoSPOptimizer](https://github.com/CGIAR-Climate-Data-Hub) archives — same chunk
geometry, codec and integer quantisation — so a cube built here matches the published
CHIRPS / CHIRTS / AgERA5 datacubes and can be sliced without reading the whole file:

```yaml
GENERAL:
  output_format: "zarr"     # "netcdf" (default) | "zarr" | "both"
  zarr_codec:    "blosc"    # "blosc" (smallest) | "zstd" (GDAL/QGIS can read it)
  zarr_quantize: true       # scaled integers | false = float32
  zarr_sharded:  false
  zarr_chunks:   null       # [time, y, x]; null = [128, 100, 100] clamped to the cube
```

```python
from aggeodata.transform.zarr_export import export_datacube_to_zarr, open_cube

export_datacube_to_zarr(cube, "climate_2020_2022.zarr")
ds = open_cube("climate_2020_2022.zarr")   # decoded to float32, CRS intact
```

**What the store contains** — `time` as int32 days since 1980-01-01, `lat`/`lon` as
float64 with CF axis attributes, a scalar `spatial_ref` carrying the CRS WKT, and one
array per variable with `scale_factor` / `add_offset` / `_FillValue`. Chunks default to
`(128, 100, 100)`, clamped down to the cube's own shape.

**Quantisation** halves the payload. Quanta follow the reference project — 0.1 for
temperature and precipitation, 10 kJ for solar radiation — all far finer than the
products' own uncertainty (ERA5 2 m temperature carries an RMSE of 1–2 K against
observations). A variable whose range would not fit its integer type falls back to
float32 automatically rather than being clipped. Set `zarr_quantize: false` for exact
float32 throughout.

**Interoperability** — measured with GDAL 3.12:

| Store | GDAL / QGIS / R `terra` |
|-------|-------------------------|
| `zarr_codec: zstd`, unsharded | reads fine, CRS intact |
| `zarr_codec: blosc` (default) | `blosc compressor not available` |
| `zarr_sharded: true` | `Unsupported codec: sharding_indexed` |

The default is `blosc` for consistency with the published archives. **Switch to
`zstd` and leave sharding off if you need to open the cube outside Python.**

---

## Use with ag-cube-cm (crop modeling)

Once you have weather and soil datacubes, pass them directly to
[ag-cube-cm](https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm) to run DSSAT pixel-by-pixel:

```bash
pip install "ag-cube-cm[models] @ git+https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm.git"
ag-cube-cm run config.yaml
```

See the [ag-cube-cm README](https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm) and the
[Spatial Crop Modeler skill](https://cgiar-climate-data-hub.github.io/skills/skills/spatial-crop-modeler/)
for a full end-to-end workflow.

---

## Repository structure

```
aggeodata/
├── src/aggeodata/
│   ├── ingestion/       # Source-specific downloaders
│   │   ├── chirps.py
│   │   ├── chirts.py
│   │   ├── zarr_backend.py  # Zarr datacube backend (CHIRPS / CHIRTS / AgERA5)
│   │   ├── agera5.py
│   │   ├── nasa_power.py
│   │   └── soil.py
│   ├── pipelines/       # High-level pipeline entry points
│   │   ├── download.py  # run_download(config_path)
│   │   └── datacube.py  # run_datacube(config_path)
│   └── transform/       # Datacube assembly
│       ├── climate_cube.py
│       ├── zarr_export.py   # Zarr v3 datacube export
│       └── soil_cube.py  # SoilDataCubeBuilder
├── options/             # Example YAML configs
└── tests/
```

---

## License

MIT — Climate Data Hub, CGIAR
