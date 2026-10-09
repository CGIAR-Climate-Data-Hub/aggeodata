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

### Faster CHIRPS / CHIRTS via Hugging Face

The UCSB servers hand out one GeoTIFF per day and throttle to a single connection,
so a multi-year request takes hours. The same archives are mirrored as consolidated
**Zarr datacubes** on Hugging Face, where a bounding box plus a date range is one
indexed read:

| Mirror | Variable in store | Dataset |
|--------|-------------------|---------|
| CHIRPS v3 daily | `precip` | [`aaguilar90/chirps-v3-daily-rnl`](https://huggingface.co/datasets/aaguilar90/chirps-v3-daily-rnl) |
| CHIRTS-ERA5 daily | `Tmax`, `Tmin` | [`aaguilar90/chirts-era5-daily`](https://huggingface.co/datasets/aaguilar90/chirts-era5-daily) |

Turn it on with one line in the config:

```yaml
GENERAL:
  use_huggingface: true    # applies to every chirps / chirts variable
```

or per variable:

```yaml
CLIMATE:
  variables:
    pr:
      source: chirps
      use_huggingface: true
    tasmax:
      source: chirts
      use_huggingface: false   # this one stays on the UCSB servers
```

Both backends write the **same per-day NetCDF layout**, so `run_datacube` is unchanged.
Requires the `download` extra (`zarr`). Notes:

* The mirrors carry CHIRPS v3 daily and CHIRTS-**ERA5** daily only. `chirts_source: chirts`
  (the original CHIRTS-daily v1.0) and `frequency="monthly"` are rejected with the
  Hugging Face backend — use the UCSB one for those.
* Temperatures in the mirror are stored rounded to 0.1 °C; the UCSB GeoTIFFs carry
  full float32 precision (max difference 0.05 °C).
* For precipitation the two backends serve **different products**: the mirror is
  CHIRPS v3 (station-blended), while this repo's UCSB v3 endpoint is
  [CHIRP v3](https://data.chc.ucsb.edu/products/CHIRP-v3.0/) (satellite-only, no
  station blending). Pick one and keep it for the whole time series.
* Don't mix backends inside one variable folder — rasterio's clip keeps one extra
  edge pixel, so the per-day grids differ. The downloader warns if it spots this.

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
        "use_huggingface": True,  # CHIRPS/CHIRTS from the HF Zarr mirrors (much faster)
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

**Read a Hugging Face mirror directly** (no files written)
```python
from aggeodata.ingestion import HF_CHIRTS_URL, open_hf_zarr, subset_hf_zarr

ds = open_hf_zarr(HF_CHIRTS_URL)
sub = subset_hf_zarr(ds, ["Tmax"], extent=[33.9, -4.7, 41.9, 5.5],
                     starting_date="2024-03-01", ending_date="2024-05-31").load()
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
│   │   ├── hf_zarr.py   # Hugging Face Zarr backend (CHIRPS / CHIRTS)
│   │   ├── agera5.py
│   │   ├── nasa_power.py
│   │   └── soil.py
│   ├── pipelines/       # High-level pipeline entry points
│   │   ├── download.py  # run_download(config_path)
│   │   └── datacube.py  # run_datacube(config_path)
│   └── transform/       # Datacube assembly
│       ├── climate_cube.py
│       └── soil_cube.py  # SoilDataCubeBuilder
├── options/             # Example YAML configs
└── tests/
```

---

## License

MIT — Climate Data Hub, CGIAR
