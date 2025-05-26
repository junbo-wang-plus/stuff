#!/usr/bin/env python3
"""
Visualize GEBCO bathymetry gradients and contours
Usage: python gebco_gradients.py <gebco_file.nc> [region]
"""
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

def calculate_gradients(elevation):
    """Calculate bathymetric gradients."""
    # Convert to numpy for gradient calculation
    elev_data = elevation.values
    
    # Calculate gradients in lat/lon directions
    dy, dx = np.gradient(elev_data)
    
    # Convert to degrees (approximate)
    lat_spacing = float(np.abs(elevation.lat[1] - elevation.lat[0]))
    lon_spacing = float(np.abs(elevation.lon[1] - elevation.lon[0]))
    
    # Convert to m/degree (rough approximation)
    dy = dy / lat_spacing  # m per degree lat
    dx = dx / lon_spacing  # m per degree lon
    
    # Calculate gradient magnitude
    gradient_mag = np.sqrt(dx**2 + dy**2)
    
    # Create xarray DataArrays with same coordinates
    gradient_mag = xr.DataArray(
        gradient_mag,
        coords=elevation.coords,
        dims=elevation.dims,
        name='gradient_magnitude'
    )
    
    return gradient_mag

def plot_bathymetry_gradients(filename, region=None):
    """Plot bathymetry with gradients and contours."""
    # Open dataset
    ds = xr.open_dataset(filename)
    
    if 'elevation' in ds.data_vars:
        elevation = ds['elevation']
    elif 'z' in ds.data_vars:
        elevation = ds['z']
    else:
        raise ValueError("Could not find elevation data variable")
    
    # Apply regional subset
    if region:
        lon_min, lon_max, lat_min, lat_max = region
        print(f"Selecting region: {region}")
        
        lat_increasing = elevation.lat[0] < elevation.lat[-1]
        
        try:
            if lat_increasing:
                elevation = elevation.sel(
                    lon=slice(lon_min, lon_max), 
                    lat=slice(lat_min, lat_max)
                )
            else:
                elevation = elevation.sel(
                    lon=slice(lon_min, lon_max), 
                    lat=slice(lat_max, lat_min)
                )
        except:
            elevation = elevation.sel(
                lon=slice(lon_min, lon_max), 
                lat=slice(lat_min, lat_max),
                method='nearest'
            )
    
    # Convert to float if needed
    if not np.issubdtype(elevation.dtype, np.floating):
        elevation = elevation.astype(float)
    
    print(f"Data shape: {elevation.shape}")
    print(f"Elevation range: {float(elevation.min())} to {float(elevation.max())} m")
    
    # Calculate gradients
    print("Calculating gradients...")
    gradient_mag = calculate_gradients(elevation)
    
    print(f"Gradient range: {float(gradient_mag.min())} to {float(gradient_mag.max())} m/degree")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Bathymetry with contours
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    
    # Custom bathymetry colormap
    ocean_colors = plt.cm.Blues_r(np.linspace(0.2, 1, 128))
    land_colors = plt.cm.terrain(np.linspace(0.25, 1, 128))
    all_colors = np.vstack([ocean_colors, land_colors])
    terrain_map = ListedColormap(all_colors)
    
    # Plot bathymetry
    im1 = elevation.plot.pcolormesh(
        ax=ax1,
        cmap=terrain_map,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        vmin=-6000,
        vmax=3000
    )
    
    # Add depth contours (only negative values - ocean)
    ocean_mask = elevation < 0
    if ocean_mask.any():
        # Depth contours every 500m and 1000m
        contour_levels = [-5000, -4000, -3000, -2000, -1000, -500, -200, -100, -50]
        cs1 = elevation.plot.contour(
            ax=ax1,
            levels=contour_levels,
            colors='white',
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            alpha=0.8
        )
        ax1.clabel(cs1, inline=True, fontsize=8, fmt='%d m')
    
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.set_title('Bathymetry with Depth Contours', fontsize=12)
    if region:
        ax1.set_extent(region, ccrs.PlateCarree())
    
    # Plot 2: Gradient magnitude
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    
    # Use log scale for gradients to show features better
    gradient_log = np.log10(gradient_mag.where(gradient_mag > 0))
    
    im2 = gradient_log.plot.pcolormesh(
        ax=ax2,
        cmap='plasma',
        transform=ccrs.PlateCarree(),
        add_colorbar=False
    )
    
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.set_title('Bathymetric Gradient Magnitude (log scale)', fontsize=12)
    if region:
        ax2.set_extent(region, ccrs.PlateCarree())
    
    # Plot 3: Steep gradients highlighted
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    
    # Highlight areas with steep gradients
    steep_threshold = np.percentile(gradient_mag.values[~np.isnan(gradient_mag.values)], 95)
    print(f"Steep gradient threshold (95th percentile): {steep_threshold:.1f} m/degree")
    
    steep_areas = gradient_mag.where(gradient_mag > steep_threshold)
    
    # Plot base bathymetry in grayscale
    elevation.plot.pcolormesh(
        ax=ax3,
        cmap='gray',
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        alpha=0.5,
        vmin=-6000,
        vmax=3000
    )
    
    # Overlay steep gradients in color
    if not np.all(np.isnan(steep_areas)):
        steep_areas.plot.pcolormesh(
            ax=ax3,
            cmap='Reds',
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            alpha=0.8
        )
    
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.set_title(f'Steep Gradients (>{steep_threshold:.0f} m/degree)', fontsize=12)
    if region:
        ax3.set_extent(region, ccrs.PlateCarree())
    
    # Plot 4: Bathymetry profile
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Create a cross-section through the middle
    if region:
        mid_lat = (lat_min + lat_max) / 2
        profile = elevation.sel(lat=mid_lat, method='nearest')
        
        ax4.plot(profile.lon, profile.values, 'b-', linewidth=2)
        ax4.fill_between(profile.lon, profile.values, 0, 
                        where=(profile.values < 0), color='lightblue', alpha=0.5)
        ax4.fill_between(profile.lon, profile.values, 0, 
                        where=(profile.values > 0), color='brown', alpha=0.5)
        
        ax4.set_xlabel('Longitude (°)')
        ax4.set_ylabel('Elevation (m)')
        ax4.set_title(f'Cross-section at {mid_lat:.1f}°N', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6, pad=0.02)
    cbar1.set_label('Elevation (m)', fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6, pad=0.02)
    cbar2.set_label('log₁₀(Gradient)', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'gebco_gradients_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    ds.close()
    
    print(f"Gradient analysis saved: {output_file}")

def plot_simple_contours(filename, region=None):
    """Simple contour plot for quick viewing."""
    ds = xr.open_dataset(filename)
    
    if 'elevation' in ds.data_vars:
        elevation = ds['elevation']
    elif 'z' in ds.data_vars:
        elevation = ds['z']
    else:
        raise ValueError("Could not find elevation data variable")
    
    # Apply regional subset
    if region:
        lon_min, lon_max, lat_min, lat_max = region
        lat_increasing = elevation.lat[0] < elevation.lat[-1]
        
        try:
            if lat_increasing:
                elevation = elevation.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            else:
                elevation = elevation.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
        except:
            elevation = elevation.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max), method='nearest')
    
    if not np.issubdtype(elevation.dtype, np.floating):
        elevation = elevation.astype(float)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot filled contours
    levels = np.arange(-6000, 3001, 250)  # Every 250m
    
    cs = elevation.plot.contourf(
        ax=ax,
        levels=levels,
        cmap='terrain',
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        extend='both'
    )
    
    # Add line contours for key depths
    key_depths = [-4000, -3000, -2000, -1000, -500, -200, -100, 0]
    elevation.plot.contour(
        ax=ax,
        levels=key_depths,
        colors='black',
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    # Map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    if region:
        ax.set_extent(region, ccrs.PlateCarree())
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title('Bathymetric Contours (250m intervals)', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    output_file = 'gebco_contours.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    ds.close()
    
    print(f"Contour plot saved: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gebco_file.nc> [region] [--simple]")
        print("Region format: lon_min,lon_max,lat_min,lat_max")
        print("--simple: Create simple contour plot only")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Parse arguments
    region = None
    simple_mode = False
    
    for arg in sys.argv[2:]:
        if arg == '--simple':
            simple_mode = True
        else:
            try:
                region = [float(x) for x in arg.split(',')]
                if len(region) != 4:
                    raise ValueError("Region must have 4 values")
            except:
                print("Invalid region format. Use: lon_min,lon_max,lat_min,lat_max")
                sys.exit(1)
    
    if simple_mode:
        print("Creating simple contour plot...")
        plot_simple_contours(filename, region)
    else:
        print("Creating detailed gradient analysis...")
        plot_bathymetry_gradients(filename, region)