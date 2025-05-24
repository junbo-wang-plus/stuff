#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
from scipy.optimize import newton
import warnings
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
    # Read wave data
    print("Loading wave data...")
    ds_waves = xr.open_dataset('waves.nc')
    
    # Try to load bathymetry from GEBCO or use simple model
    try:
        # Try different possible bathymetry sources
        if 'deptho' in ds_waves.data_vars:
            print("Found bathymetry in wave data")
            bathymetry = ds_waves['deptho']
        else:
            print("Loading GEBCO bathymetry...")
            # Try common GEBCO filenames
            try:
                ds_bathy = xr.open_dataset('GEBCO_2024.nc')
                bathymetry = ds_bathy.elevation
            except FileNotFoundError:
                ds_bathy = xr.open_dataset('gebco_2024_n65.0_s30.0_w-55.0_e-15.0.nc')
                bathymetry = ds_bathy.elevation
            
            # Interpolate to wave grid
            bathymetry = bathymetry.interp(
                lat=ds_waves.latitude, 
                lon=ds_waves.longitude
            )
            ds_bathy.close()
    except (FileNotFoundError, KeyError):
        print("Bathymetry file not found, using constant depth model")
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
    
    # Create static plot of kh
    print("Creating static plot...")
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    kh_plot = ds_output.dimensionless_depth.isel(time=0).plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), levels=20, cmap='plasma',
        add_colorbar=False
    )
    plt.colorbar(kh_plot, ax=ax, shrink=0.7, label='Dimensionless depth (kh)')
    
    ax.set_title(f'Dimensionless water depth (kh) at {ds_waves.time.values[0]}')
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kh_static.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved static plot: kh_static.png")
    
    # Create video of kh time series
    print("Creating kh video...")
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
        
        time_str = str(ds_waves.time.values[t])[:19]
        ax.set_title(f'Dimensionless water depth (kh) at {time_str}')
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        return [cs]
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(ds_waves.time), blit=False)
    ani.save('kh_timeseries.mp4', fps=3, extra_args=['-vcodec', 'libx264'], dpi=150)
    plt.close()
    
    print("Saved video: kh_timeseries.mp4")
    
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
