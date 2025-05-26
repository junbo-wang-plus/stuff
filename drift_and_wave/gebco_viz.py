#!/usr/bin/env python3
"""
Visualize GEBCO bathymetry data
Usage: python visualize_gebco.py <gebco_file.nc> [region]
"""
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def inspect_gebco_region(filename, region):
    """Quick inspection of GEBCO data in specified region."""
    ds = xr.open_dataset(filename)
    elevation = ds['elevation'] if 'elevation' in ds.data_vars else ds['z']
    
    if region:
        lon_min, lon_max, lat_min, lat_max = region
        lat_increasing = elevation.lat[0] < elevation.lat[-1]
        
        if lat_increasing:
            subset = elevation.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        else:
            subset = elevation.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
        
        print(f"Regional subset:")
        print(f"  Shape: {subset.shape}")
        print(f"  Valid points: {(~np.isnan(subset)).sum().item()}")
        print(f"  Data range: {float(subset.min())} to {float(subset.max())}")
        print(f"  Ocean depth range: {float(subset.where(subset < 0).min())} to {float(subset.where(subset < 0).max())}")
        print(f"  Land elevation range: {float(subset.where(subset > 0).min())} to {float(subset.where(subset > 0).max())}")
    
    ds.close()

def visualize_gebco(filename, region=None, output_file='gebco_bathymetry.png'):
    """
    Visualize GEBCO bathymetry data.
    
    Parameters:
    -----------
    filename : str
        Path to GEBCO NetCDF file
    region : list, optional
        [lon_min, lon_max, lat_min, lat_max] for regional subset
    output_file : str
        Output filename for the plot
    """
    # Open GEBCO dataset
    ds = xr.open_dataset(filename)
    
    # GEBCO uses 'elevation' variable name
    if 'elevation' in ds.data_vars:
        elevation = ds['elevation']
    elif 'z' in ds.data_vars:
        elevation = ds['z']
    else:
        # Check available variables
        print(f"Available variables: {list(ds.data_vars)}")
        raise ValueError("Could not find elevation data variable")
    
    # Apply regional subset if specified
    if region:
        lon_min, lon_max, lat_min, lat_max = region
        print(f"Selecting region: lon {lon_min} to {lon_max}, lat {lat_min} to {lat_max}")
        
        # Check coordinate order and ranges
        lat_increasing = elevation.lat[0] < elevation.lat[-1]
        print(f"Latitude increasing: {lat_increasing}")
        print(f"Available lon range: {float(elevation.lon.min())} to {float(elevation.lon.max())}")
        print(f"Available lat range: {float(elevation.lat.min())} to {float(elevation.lat.max())}")
        
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
        except Exception as e:
            print(f"Regional selection failed: {e}")
            print("Using nearest neighbor selection instead...")
            elevation = elevation.sel(
                lon=slice(lon_min, lon_max), 
                lat=slice(lat_min, lat_max),
                method='nearest'
            )
    
    # Check data properties
    print(f"Data shape: {elevation.shape}")
    print(f"Data type: {elevation.dtype}")
    
    # Convert to float if needed
    if not np.issubdtype(elevation.dtype, np.floating):
        print("Converting data to float...")
        elevation = elevation.astype(float)
    
    print(f"Data range: {float(elevation.min())} to {float(elevation.max())}")
    print(f"Number of valid points: {(~np.isnan(elevation)).sum().item()}")
    
    # Handle case where all data is NaN or masked
    if np.all(np.isnan(elevation)):
        print("Warning: All data is NaN, cannot plot")
        return
    
    # Create figure with cartographic projection
    fig, ax = plt.subplots(figsize=(12, 8), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Define bathymetry colormap
    # Blue-white-brown: deep ocean -> shallow -> land
    from matplotlib.colors import ListedColormap
    
    # Create custom colormap for bathymetry
    ocean_colors = plt.cm.Blues_r(np.linspace(0.2, 1, 128))
    land_colors = plt.cm.terrain(np.linspace(0.25, 1, 128))
    all_colors = np.vstack([ocean_colors, land_colors])
    terrain_map = ListedColormap(all_colors)
    
    # Plot bathymetry/topography
    try:
        im = elevation.plot.contourf(
            ax=ax, 
            levels=50,
            cmap=terrain_map,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            vmin=-6000,  # Deep ocean
            vmax=3000    # High mountains
        )
    except (TypeError, ValueError) as e:
        print(f"Contourf failed: {e}")
        print("Trying alternative plotting method...")
        
        # Alternative: use pcolormesh
        im = elevation.plot.pcolormesh(
            ax=ax,
            cmap=terrain_map,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            vmin=-6000,
            vmax=3000
        )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
    
    # Set extent if region specified
    if region:
        ax.set_extent(region, ccrs.PlateCarree())
    else:
        ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Title
    ax.set_title('GEBCO Bathymetry and Topography', fontsize=14, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Close dataset
    ds.close()
    
    print(f"Bathymetry visualization saved: {output_file}")

def print_gebco_info(filename):
    """Print basic information about GEBCO file."""
    ds = xr.open_dataset(filename)
    
    print(f"GEBCO file: {filename}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")
    
    # Check coordinate names (can vary)
    coord_names = list(ds.coords)
    print(f"Coordinates: {coord_names}")
    
    # Spatial extent
    if 'lon' in ds.coords:
        lon_coord = 'lon'
    elif 'longitude' in ds.coords:
        lon_coord = 'longitude'
    else:
        lon_coord = None
        
    if 'lat' in ds.coords:
        lat_coord = 'lat'
    elif 'latitude' in ds.coords:
        lat_coord = 'latitude'
    else:
        lat_coord = None
    
    if lon_coord and lat_coord:
        lon_range = (float(ds[lon_coord].min()), float(ds[lon_coord].max()))
        lat_range = (float(ds[lat_coord].min()), float(ds[lat_coord].max()))
        print(f"Longitude range: {lon_range}")
        print(f"Latitude range: {lat_range}")
    
    ds.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gebco_file.nc> [region]")
        print("Region format: lon_min,lon_max,lat_min,lat_max")
        print("Example: python visualize_gebco.py GEBCO_2024.nc -10,5,50,70")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Parse region if provided
    region = None
    if len(sys.argv) > 2:
        try:
            region = [float(x) for x in sys.argv[2].split(',')]
            if len(region) != 4:
                raise ValueError("Region must have 4 values")
        except:
            print("Invalid region format. Use: lon_min,lon_max,lat_min,lat_max")
            sys.exit(1)
    
    # Print file info
    print_gebco_info(filename)
    print()
    
    # Inspect region if specified
    if region:
        inspect_gebco_region(filename, region)
        print()
    
    # Create visualization
    if region:
        output_name = 'gebco_bathymetry_regional.png'
        print(f"Creating regional plot for: {region}")
    else:
        output_name = 'gebco_bathymetry_global.png'
        print("Creating global plot...")
    
    visualize_gebco(filename, region, output_name)