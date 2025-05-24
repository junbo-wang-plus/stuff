#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
from scipy.optimize import newton
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

def solve_dispersion_relation_vectorized(omega, h, g=9.81):
    """
    Solve the dispersion relation: omega^2 = g*k*tanh(k*h)
    for wavenumber k given angular frequency omega and water depth h
    Uses vectorized Newton's method for efficiency
    """
    # Initial guess using deep water approximation
    k = omega**2 / g
    
    # Newton's method iteration
    for _ in range(10):  # Usually converges in 3-4 iterations
        tanh_kh = np.tanh(k * h)
        sech2_kh = 1 - tanh_kh**2
        
        f = omega**2 - g * k * tanh_kh
        df = -g * (tanh_kh + k * h * sech2_kh)
        
        # Avoid division by zero
        mask = np.abs(df) > 1e-12
        k[mask] = k[mask] - f[mask] / df[mask]
    
    return k

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze wave data and compute dispersion relation parameters')
    parser.add_argument('wave_file', help='Wave forecast NetCDF file (e.g., waves.nc)')
    parser.add_argument('--bathy', '--bathymetry', dest='bathy_file', 
                       help='Bathymetry NetCDF file (optional, will use constant depth if not provided)')
    
    args = parser.parse_args()
    
    # Read wave data
    print(f"Loading wave data from {args.wave_file}...")
    try:
        ds_waves = xr.open_dataset(args.wave_file)
    except FileNotFoundError:
        print(f"Error: Wave file '{args.wave_file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading wave file: {e}")
        sys.exit(1)
    
    # Load bathymetry
    bathymetry = None
    if args.bathy_file:
        print(f"Loading bathymetry from {args.bathy_file}...")
        try:
            ds_bathy = xr.open_dataset(args.bathy_file)
            # Try common variable names for bathymetry/elevation
            if 'elevation' in ds_bathy.data_vars:
                bathymetry = ds_bathy.elevation
            elif 'deptho' in ds_bathy.data_vars:
                bathymetry = ds_bathy.deptho  
            elif 'depth' in ds_bathy.data_vars:
                bathymetry = ds_bathy.depth
            else:
                available_vars = list(ds_bathy.data_vars)
                print(f"Unknown bathymetry variable. Available variables: {available_vars}")
                print("Please check your bathymetry file format")
                sys.exit(1)
            
            # Interpolate to wave grid if coordinates don't match
            try:
                # Try different coordinate names
                if 'lat' in ds_bathy.dims and 'lon' in ds_bathy.dims:
                    bathymetry = bathymetry.interp(lat=ds_waves.latitude, lon=ds_waves.longitude)
                elif 'latitude' in ds_bathy.dims and 'longitude' in ds_bathy.dims:
                    bathymetry = bathymetry.interp(latitude=ds_waves.latitude, longitude=ds_waves.longitude)
                else:
                    print("Grid interpolation not needed or coordinates already match")
            except Exception as e:
                print(f"Warning: Could not interpolate bathymetry grid: {e}")
                print("Using bathymetry data as-is")
            
            ds_bathy.close()
            print("Bathymetry loaded successfully")
            
        except FileNotFoundError:
            print(f"Warning: Bathymetry file '{args.bathy_file}' not found!")
            bathymetry = None
        except Exception as e:
            print(f"Warning: Error loading bathymetry file: {e}")
            bathymetry = None
    
    # Check if bathymetry is embedded in wave data
    if bathymetry is None and 'deptho' in ds_waves.data_vars:
        print("Found bathymetry in wave data file")
        bathymetry = ds_waves['deptho']
    
    # Use constant depth as fallback
    if bathymetry is None:
        print("Using constant depth model (1000m)")
        bathymetry = xr.DataArray(
            -1000 * np.ones((len(ds_waves.latitude), len(ds_waves.longitude))),
            coords={'latitude': ds_waves.latitude, 'longitude': ds_waves.longitude},
            dims=['latitude', 'longitude']
        )
    
    # Get wave parameters
    peak_period = ds_waves['VTPK']  # Wave period at spectral peak (s)
    
    # Use peak direction if available, otherwise mean direction
    if 'VPED' in ds_waves.data_vars:
        wave_direction = ds_waves['VPED']
        print("Using peak direction (VPED)")
    else:
        wave_direction = ds_waves['VMDR']
        print("Using mean direction (VMDR)")
    
    print(f"Processing {len(ds_waves.time)} time steps...")
    
    # Compute derived parameters
    peak_frequency = 1.0 / peak_period
    omega = 2 * np.pi * peak_frequency
    
    # Water depth (positive values) - only where bathymetry is negative
    water_depth = xr.where(bathymetry < 0, -bathymetry, np.nan)
    
    # Create ocean mask for wave calculations
    ocean_mask = bathymetry < 0
    
    # Solve dispersion relation only in ocean areas
    print("Solving dispersion relation...")
    wavenumber_magnitude = xr.where(
        ocean_mask,
        xr.apply_ufunc(
            solve_dispersion_relation_vectorized,
            omega, water_depth,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            dask='allowed'
        ),
        np.nan
    )
    
    # Convert direction from "from" to "to" (propagation direction)
    direction_to = wave_direction + 180
    direction_rad = np.deg2rad(direction_to)
    
    # Compute wavenumber components (x=eastward, y=northward) only in ocean
    kx = xr.where(ocean_mask, wavenumber_magnitude * np.sin(direction_rad), np.nan)
    ky = xr.where(ocean_mask, wavenumber_magnitude * np.cos(direction_rad), np.nan)
    
    # Compute dimensionless water depth kh only in ocean
    kh = xr.where(ocean_mask, wavenumber_magnitude * water_depth, np.nan)
    
    # Create output dataset
    print("Creating output dataset...")
    ds_output = xr.Dataset({
        'peak_frequency': peak_frequency,
        'wavenumber_magnitude': wavenumber_magnitude,
        'wavenumber_x': kx,
        'wavenumber_y': ky,
        'dimensionless_depth': kh,
        'water_depth': water_depth,
        'ocean_mask': ocean_mask,
        'peak_period': peak_period,
        'wave_direction': wave_direction,
    })
    
    # Add attributes
    ds_output['peak_frequency'].attrs = {'long_name': 'Peak frequency', 'units': 'Hz'}
    ds_output['wavenumber_magnitude'].attrs = {'long_name': 'Wavenumber magnitude', 'units': 'm^-1'}
    ds_output['wavenumber_x'].attrs = {'long_name': 'Wavenumber x-component (eastward)', 'units': 'm^-1'}
    ds_output['wavenumber_y'].attrs = {'long_name': 'Wavenumber y-component (northward)', 'units': 'm^-1'}
    ds_output['dimensionless_depth'].attrs = {'long_name': 'Dimensionless water depth (kh)', 'units': 'dimensionless'}
    ds_output['water_depth'].attrs = {'long_name': 'Water depth', 'units': 'm'}
    ds_output['ocean_mask'].attrs = {'long_name': 'Ocean mask (True=ocean, False=land)', 'units': 'boolean'}
    
    # Save output
    output_filename = 'wave_analysis.nc'
    ds_output.to_netcdf(output_filename)
    print(f"Saved analysis to {output_filename}")
    
    # Determine arrow subsampling stride based on grid size (used for all plots)
    stride_lon = max(1, len(ds_waves.longitude) // 25)
    stride_lat = max(1, len(ds_waves.latitude) // 25)
    
    # Create static plot of kh
    print("Creating static plot with direction arrows...")
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    kh_plot = ds_output.dimensionless_depth.isel(time=0).plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), levels=20, cmap='plasma',
        add_colorbar=False
    )
    plt.colorbar(kh_plot, ax=ax, shrink=0.7, label='Dimensionless depth (kh)')
    
    # Add wave direction arrows to static plot
    wave_dir_static = wave_direction.isel(time=0)
    dir_to_static = wave_dir_static + 180
    dir_rad_static = np.deg2rad(dir_to_static)
    
    u_static = np.sin(dir_rad_static)
    v_static = np.cos(dir_rad_static)
    
    # Subsample arrows and only show over ocean
    kh_static = ds_output.dimensionless_depth.isel(time=0)
    lon_sub = kh_static.longitude[::stride_lon]
    lat_sub = kh_static.latitude[::stride_lat]
    u_sub = u_static[::stride_lat, ::stride_lon]
    v_sub = v_static[::stride_lat, ::stride_lon]
    kh_sub = kh_static[::stride_lat, ::stride_lon]
    
    valid_mask = ~np.isnan(kh_sub)
    if valid_mask.any():
        ax.quiver(lon_sub, lat_sub, u_sub, v_sub, 
                 transform=ccrs.PlateCarree(),
                 scale=30, width=0.003, color='white', alpha=0.8,
                 headwidth=3, headlength=4)
    
    ax.set_title(f'Dimensionless water depth (kh) with wave directions at {ds_waves.time.values[0]}')
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kh_static.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved static plot: kh_static.png")
    
    # Create video of kh time series
    print("Creating kh video with direction arrows...")
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    kh_max = float(ds_output.dimensionless_depth.max())
    levels = np.linspace(0, kh_max, 21)
    
    # Create persistent colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=kh_max)
    sm = ScalarMappable(norm=norm, cmap='plasma')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, label='Dimensionless depth (kh)')
    
    def update_frame(t):
        ax.clear()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        
        kh_data = ds_output.dimensionless_depth.isel(time=t)
        cs = ax.contourf(kh_data.longitude, kh_data.latitude, kh_data,
                        levels=levels, cmap='plasma', norm=norm,
                        transform=ccrs.PlateCarree())
        
        # Add wave direction arrows
        wave_dir = wave_direction.isel(time=t)
        # Convert from "direction from" to "direction to" (propagation direction)
        dir_to = wave_dir + 180
        dir_rad = np.deg2rad(dir_to)
        
        # Calculate arrow components (u=eastward, v=northward)
        u_arrows = np.sin(dir_rad)
        v_arrows = np.cos(dir_rad)
        
        # Subsample for arrow display and only show over ocean
        lon_sub = kh_data.longitude[::stride_lon]
        lat_sub = kh_data.latitude[::stride_lat]
        u_sub = u_arrows[::stride_lat, ::stride_lon]
        v_sub = v_arrows[::stride_lat, ::stride_lon]
        kh_sub = kh_data[::stride_lat, ::stride_lon]
        
        # Only show arrows where kh is valid (over ocean)
        valid_mask = ~np.isnan(kh_sub)
        
        if valid_mask.any():
            ax.quiver(lon_sub, lat_sub, u_sub, v_sub, 
                     transform=ccrs.PlateCarree(),
                     scale=30, width=0.003, color='white', alpha=0.8,
                     headwidth=3, headlength=4)
        
        time_str = str(ds_waves.time.values[t])[:19]
        ax.set_title(f'Dimensionless water depth (kh) with wave directions at {time_str}')
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        return [cs]
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(ds_waves.time), blit=False)
    ani.save('kh_timeseries.mp4', fps=3, extra_args=['-vcodec', 'libx264'], dpi=150)
    plt.close()
    
    print("Saved video: kh_timeseries.mp4")
    
    # Create log-scale video of kh time series
    print("Creating kh log-scale video with direction arrows...")
    fig_log = plt.figure(figsize=(15, 8))
    ax_log = plt.axes(projection=ccrs.PlateCarree())
    
    # Use log scale for better visualization of wave regimes
    kh_min = float(ds_output.dimensionless_depth.where(ds_output.dimensionless_depth > 0).min())
    kh_max = float(ds_output.dimensionless_depth.max())
    
    # Create log-spaced levels with wave regime boundaries
    import matplotlib.colors as colors
    from matplotlib.ticker import LogFormatter
    
    # Set reasonable bounds for log scale
    log_min = max(0.01, kh_min)  # Avoid log(0)
    log_max = kh_max
    
    norm_log = colors.LogNorm(vmin=log_min, vmax=log_max)
    levels_log = np.logspace(np.log10(log_min), np.log10(log_max), 21)
    
    # Create persistent log colorbar
    sm_log = ScalarMappable(norm=norm_log, cmap='plasma')
    sm_log.set_array([])
    cbar_log = plt.colorbar(sm_log, ax=ax_log, shrink=0.7, label='Dimensionless depth (kh) - Log scale')
    
    # Add regime boundary lines to colorbar
    shallow_boundary = np.pi/10  # ~0.31
    deep_boundary = np.pi        # ~3.14
    
    if log_min <= shallow_boundary <= log_max:
        cbar_log.ax.axhline(shallow_boundary, color='red', linestyle='--', linewidth=1)
        cbar_log.ax.text(0.5, shallow_boundary, 'Shallow', transform=cbar_log.ax.get_yaxis_transform(), 
                        ha='left', va='bottom', color='red', fontsize=8)
    
    if log_min <= deep_boundary <= log_max:
        cbar_log.ax.axhline(deep_boundary, color='blue', linestyle='--', linewidth=1)
        cbar_log.ax.text(0.5, deep_boundary, 'Deep', transform=cbar_log.ax.get_yaxis_transform(), 
                        ha='left', va='bottom', color='blue', fontsize=8)
    
    def update_frame_log(t):
        ax_log.clear()
        ax_log.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_log.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        
        kh_data = ds_output.dimensionless_depth.isel(time=t)
        # Mask values that are too small for log scale
        kh_masked = xr.where(kh_data >= log_min, kh_data, np.nan)
        
        cs_log = ax_log.contourf(kh_masked.longitude, kh_masked.latitude, kh_masked,
                               levels=levels_log, cmap='plasma', norm=norm_log,
                               transform=ccrs.PlateCarree())
        
        # Add wave direction arrows
        wave_dir = wave_direction.isel(time=t)
        # Convert from "direction from" to "direction to" (propagation direction)
        dir_to = wave_dir + 180
        dir_rad = np.deg2rad(dir_to)
        
        # Calculate arrow components (u=eastward, v=northward)
        u_arrows = np.sin(dir_rad)
        v_arrows = np.cos(dir_rad)
        
        # Subsample for arrow display and only show over ocean
        lon_sub = kh_masked.longitude[::stride_lon]
        lat_sub = kh_masked.latitude[::stride_lat]
        u_sub = u_arrows[::stride_lat, ::stride_lon]
        v_sub = v_arrows[::stride_lat, ::stride_lon]
        kh_sub = kh_masked[::stride_lat, ::stride_lon]
        
        # Only show arrows where kh is valid (over ocean)
        valid_mask = ~np.isnan(kh_sub)
        
        if valid_mask.any():
            ax_log.quiver(lon_sub, lat_sub, u_sub, v_sub, 
                         transform=ccrs.PlateCarree(),
                         scale=30, width=0.003, color='white', alpha=0.8,
                         headwidth=3, headlength=4)
        
        time_str = str(ds_waves.time.values[t])[:19]
        ax_log.set_title(f'Dimensionless water depth (kh) - Log scale with wave directions at {time_str}')
        ax_log.gridlines(draw_labels=True, alpha=0.3)
        
        return [cs_log]
    
    ani_log = animation.FuncAnimation(fig_log, update_frame_log, frames=len(ds_waves.time), blit=False)
    ani_log.save('kh_timeseries_log.mp4', fps=3, extra_args=['-vcodec', 'libx264'], dpi=150)
    plt.close()
    
    print("Saved log-scale video: kh_timeseries_log.mp4")
    
    # Print summary statistics
    print("\nSummary statistics:")
    ocean_points = ocean_mask.sum().values
    total_points = ocean_mask.size
    print(f"Ocean coverage: {ocean_points}/{total_points} points ({100*ocean_points/total_points:.1f}%)")
    
    # Statistics only for ocean areas
    valid_freq = peak_frequency.where(ocean_mask)
    valid_k = wavenumber_magnitude.where(ocean_mask)
    valid_kh = kh.where(ocean_mask)
    
    print(f"Peak frequency range: {valid_freq.min().values:.4f} - {valid_freq.max().values:.4f} Hz")
    print(f"Wavenumber range: {valid_k.min().values:.4f} - {valid_k.max().values:.4f} m⁻¹")
    print(f"kh range: {valid_kh.min().values:.2f} - {valid_kh.max().values:.2f}")
    print(f"Water depth range: {water_depth.min().values:.1f} - {water_depth.max().values:.1f} m")
    
    ds_waves.close()
    ds_output.close()
    print("Done!")

if __name__ == "__main__":
    main()