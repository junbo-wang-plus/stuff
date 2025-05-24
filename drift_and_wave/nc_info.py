#!/usr/bin/env python3
import sys
import xarray as xr
from datetime import datetime

def print_nc_info(filename):
    """Print basic information about a NetCDF file."""
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(filename)
        
        # Print basic file information
        print(f"File: {filename}")
        
        # Variables
        print("\nVariables:")
        for var_name in ds.variables:
            var = ds[var_name]
            shape_str = "×".join(str(s) for s in var.shape)
            print(f"  {var_name}: {shape_str}")
        
        # Time range (if present)
        if 'time' in ds.dims:
            try:
                start_time = ds.time.values[0]
                end_time = ds.time.values[-1]
                print(f"\nTime range: {start_time} to {end_time}")
                print(f"Time steps: {len(ds.time)}")
            except:
                print("\nTime dimension exists but couldn't extract values")
        
        # Spatial dimensions (if present)
        spatial_dims = ['latitude', 'longitude', 'lat', 'lon', 'x', 'y']
        print("\nSpatial coverage:")
        for dim in spatial_dims:
            if dim in ds.dims:
                print(f"  {dim}: {len(ds[dim])} points")
                if hasattr(ds[dim], 'min') and hasattr(ds[dim], 'max'):
                    print(f"    range: {ds[dim].values.min()} to {ds[dim].values.max()}")
        
        # Close the dataset
        ds.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <netcdf_filename>")
        sys.exit(1)
    
    print_nc_info(sys.argv[1])
