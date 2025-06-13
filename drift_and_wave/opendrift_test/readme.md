# OpenDrift Trajectory & Wave Analysis Project

## Overview
This project calculates particle trajectories using OpenDrift with external oceanographic datasets and performs wave-bathymetry analysis for surface gravity wave dynamics.

## Prerequisites

### Environment Setup
```bash
eval "$(/Users/jwang/miniforge3/bin/conda shell.zsh hook)"
conda activate opendrift
```

### Dependencies
```bash
pip install opendrift xarray matplotlib cartopy netcdf4 copernicusmarine ecmwf-opendata scipy cfgrib
```

### Required Datasets
**Pre-download manually:**
- GEBCO bathymetry data from [GEBCO](https://www.gebco.net/) → `GEBCO_2024.nc`
  - Latest: GEBCO_2024 Grid (published July 2024)
  - Global 15 arc-second resolution (~450m spacing)

**Downloaded by scripts:**
- **CMEMS ocean data** (requires free registration at [marine.copernicus.eu](https://marine.copernicus.eu))
  - Sign Service Level Agreement for full access
  - Forecast range: **10 days** (240 hours)
  - Update frequency: Twice daily at 00:00/12:00 UTC
  - Archive: 2-year rolling archive available
- **ECMWF wind forecasts** (free, no registration)
  - HRES: **10 days** deterministic forecast
  - ENS: **15 days** ensemble forecast  
  - Update frequency: Every 6-12 hours (00Z, 06Z, 12Z, 18Z)
  - Open data: 0.25° resolution, 3-hourly, CC-BY-4.0 license

## Part 1: OpenDrift Trajectory Calculations

### Data Download
Download required datasets from CMEMS and ECMWF:

```bash
# Set CMEMS credentials (required)
# Update username/password in download scripts before running

# Ocean currents, tides, and waves (CMEMS)
python download.py  # → currents_tides.nc, waves.nc

# Wind data (ECMWF - no credentials needed)
python download_wind.py  # → wind.grib2
python ecmwf_convert.py  # → wind.nc

# Temperature data (optional)
python download_temp.py  # → sst_forecast.nc
```

**Note:** CMEMS has a 2GB limit per download request. For larger datasets, split into multiple time periods.

**Datasets used:**
- `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i` - Ocean currents and tides (uo, vo, utide, vtide)
  - Resolution: 1/12° (~8km), hourly
- `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i` - Wave data (VHM0, VMDR, VTPK, VSDX, VSDY)  
  - Resolution: 1/12° (~8km), 3-hourly
- `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m` - Sea surface temperature (thetao)
  - Resolution: 1/12° (~8km), daily
- ECMWF wind forecasts (u10, v10 → eastward_wind, northward_wind)
  - Resolution: 0.25° (~25km), 3-hourly

### Run Trajectories
```bash
# Currents + waves vs currents only
python run.py  # → drift_comp.png, drift.mp4

# With wind forcing
python run_with_wind.py  # → drift_with_wind.png, drift_with_wind.mp4
```

### Visualization
```bash
# Static plots
python visualize.py  # → currents_tides.png, waves.png

# Videos  
python visualize_video.py  # → currents_video.mp4, waves_video.mp4

# Wind visualization
python visualize_grib.py wind.grib2  # → wind.png, wind_video.mp4

# Temperature analysis
python visualize_temperature.py  # → multiple temperature plots/videos
```

## Part 2: Wave-Bathymetry Analysis

### Bathymetry Data
Download GEBCO bathymetry data manually from [GEBCO](https://www.gebco.net/).

### Wave Dispersion Analysis
Solve the dispersion relation ω² = gk·tanh(kh) and compute wave parameters:

```bash
# With bathymetry
python wave_analysis.py waves.nc --bathy GEBCO_2024.nc

# Without bathymetry (constant depth)
python wave_analysis.py waves.nc
```

**Outputs:**
- `wave_analysis.nc` - Computed wavenumber components, dimensionless depth (kh)
- `kh_static.png` - Static plot of kh with wave directions
- `kh_timeseries.mp4` - Time evolution of kh
- `kh_timeseries_log.mp4` - Log-scale version showing wave regimes

### Bathymetry Visualization
```bash
# Basic bathymetry
python gebco_viz.py GEBCO_2024.nc [lon_min,lon_max,lat_min,lat_max]  
# → gebco_bathymetry_regional.png or gebco_bathymetry_global.png

# Gradient analysis  
python gebco_gradient_viz.py GEBCO_2024.nc [region] [--simple]
# → gebco_gradients_analysis.png or gebco_contours.png (--simple)
```

## Key Variables

### CMEMS Wave Data (GLOBAL_ANALYSISFORECAST_WAV_001_027)
- `VHM0` - Significant wave height (m)
- `VTPK` - Peak wave period (s) 
- `VMDR` - Mean wave direction (°)
- `VSDX/VSDY` - Stokes drift components (m/s)

### Current Data 
- `uo/vo` - Ocean velocity components (m/s)
- `utide/vtide` - Tidal velocity components (m/s)

### GEBCO Bathymetry
- `elevation` - Elevation/depth values (m, negative = ocean depth, positive = land elevation)
- Uses WGS84 coordinate system, pixel-center registered
- Type Identifier (TID) grid available showing data source quality

## Region Configuration
Default study area: North Atlantic
- Longitude: -55° to -15°
- Latitude: 35° to 65°

Edit region bounds in download scripts as needed.

## Notes
- CMEMS requires registration and credentials
- ECMWF data is freely available
- Wave analysis computes shallow/intermediate/deep water wave regimes based on kh values:
  - Shallow water: kh < π/10 (~0.31)
  - Intermediate water: π/10 < kh < π (~3.14)  
  - Deep water: kh > π
- Trajectory comparisons show effects of waves, tides, and wind forcing
- All data uses WGS84 coordinate system
