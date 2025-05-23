#!/usr/bin/env python3
"""
Minimal GRIB2 wind visualization script.
Outputs: variable info, static image, and video.
"""

import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation

# Fixed region coordinates
REGION = [-55.0, -15.0, 35.0, 65.0]  # [lon_min, lon_max, lat_min, lat_max]

def print_variables(filename):
    """Print basic variable information."""
    ds = xr.open_dataset(filename, engine='cfgrib')
    print(f"Variables in {filename}:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        shape_str = "×".join(str(s) for s in var.shape)
        print(f"  {var_name}: {shape_str}")
        if hasattr(var, 'long_name'):
            print(f"    {var.long_name}")
    
    if 'step' in ds.dims:
        print(f"Forecast steps: {len(ds.step)} ({ds.step.values[0]} to {ds.step.values[-1]})")
    ds.close()

def create_static_plot(filename, use_region=True, output_file='wind.png'):
    """Create static wind plot."""
    ds = xr.open_dataset(filename, engine='cfgrib')
    
    # Get wind components
    u, v = ds['u10'].isel(step=0), ds['v10'].isel(step=0)
    
    # Apply region if requested
    if use_region:
        lon_min, lon_max, lat_min, lat_max = REGION
        u = u.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        v = v.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
    
    wind_speed = np.sqrt(u**2 + v**2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    # Plot wind speed and vectors
    speed_plot = ax.contourf(u.longitude, u.latitude, wind_speed, 
                           levels=20, cmap='viridis', transform=ccrs.PlateCarree())
    plt.colorbar(speed_plot, ax=ax, shrink=0.7, label='Wind Speed (m/s)')
    
    stride = max(1, len(u.longitude) // 20)
    ax.quiver(u.longitude[::stride], u.latitude[::stride],
             u.values[::stride, ::stride], v.values[::stride, ::stride],
             transform=ccrs.PlateCarree(), scale=200, width=0.003, color='white', alpha=0.7)
    
    if use_region:
        ax.set_extent(REGION, ccrs.PlateCarree())
    
    ax.set_title(f'ECMWF 10m Wind - Step {ds.step.values[0]}')
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    ds.close()
    print(f"Static plot saved: {output_file}")

def create_video(filename, use_region=True, output_file='wind_video.mp4'):
    """Create wind animation video."""
    ds = xr.open_dataset(filename, engine='cfgrib')
    
    # Get wind components for all steps
    u_all, v_all = ds['u10'], ds['v10']
    
    # Apply region if requested
    if use_region:
        lon_min, lon_max, lat_min, lat_max = REGION
        u_all = u_all.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        v_all = v_all.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
    
    # Calculate max speed for consistent color scale
    speed_max = 0
    for step in range(len(ds.step)):
        u, v = u_all.isel(step=step), v_all.isel(step=step)
        speed = np.sqrt(u**2 + v**2)
        speed_max = max(speed_max, float(speed.max()))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    def update_frame(step):
        ax.clear()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        
        u, v = u_all.isel(step=step), v_all.isel(step=step)
        wind_speed = np.sqrt(u**2 + v**2)
        
        # Plot with fixed color scale
        speed_plot = ax.contourf(u.longitude, u.latitude, wind_speed,
                               levels=20, cmap='viridis', transform=ccrs.PlateCarree(),
                               vmin=0, vmax=speed_max)
        
        stride = max(1, len(u.longitude) // 20)
        ax.quiver(u.longitude[::stride], u.latitude[::stride],
                 u.values[::stride, ::stride], v.values[::stride, ::stride],
                 transform=ccrs.PlateCarree(), scale=200, width=0.003, color='white', alpha=0.7)
        
        if use_region:
            ax.set_extent(REGION, ccrs.PlateCarree())
        
        ax.set_title(f'ECMWF 10m Wind - Step {ds.step.values[step]}')
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        return [ax]
    
    # Create and save animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(ds.step), blit=False)
    ani.save(output_file, fps=3, extra_args=['-vcodec', 'libx264'], dpi=150)
    plt.close()
    ds.close()
    print(f"Video saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <grib_file> [global]")
        print("  Default: regional plot")
        print("  Add 'global' for global plot")
        sys.exit(1)
    
    filename = sys.argv[1]
    use_region = 'global' not in sys.argv
    
    print_variables(filename)
    print("\nCreating static plot...")
    create_static_plot(filename, use_region)
    print("Creating video...")
    create_video(filename, use_region)
    print("Done.")

if __name__ == "__main__":
    main()
