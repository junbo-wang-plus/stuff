#!/usr/bin/env python3
"""
Targeted converter for ECMWF GRIB2 wind data with time+step structure
"""
import xarray as xr
import numpy as np
import pandas as pd

def convert_ecmwf_wind():
    print("Converting ECMWF wind.grib2 to wind.nc...")
    
    # Open the GRIB file
    ds = xr.open_dataset('wind.grib2', engine='cfgrib')
    
    print(f"Original variables: {list(ds.data_vars)}")
    print(f"Original dimensions: {dict(ds.dims)}")
    print(f"Time: {ds.time.values}")
    print(f"Steps: {len(ds.step)} forecast steps")
    
    # Convert the time+step structure to proper time series
    base_time = pd.to_datetime(ds.time.values)
    steps = pd.to_timedelta(ds.step.values)
    
    # Create proper time coordinate
    valid_times = [base_time + step for step in steps]
    
    print(f"Creating time series from {valid_times[0]} to {valid_times[-1]}")
    
    # Create new coordinates dictionary
    new_coords = {}
    for coord_name, coord_var in ds.coords.items():
        if coord_name == 'step':
            # Replace step with proper time coordinate
            new_coords['time'] = ('time', valid_times)
        elif coord_name != 'time':  # Skip the original scalar time
            new_coords[coord_name] = coord_var
    
    # Create new data variables with renamed dimensions
    new_data_vars = {}
    for var_name, var_data in ds.data_vars.items():
        # Replace 'step' dimension with 'time' in the variable
        new_dims = []
        for dim in var_data.dims:
            if dim == 'step':
                new_dims.append('time')
            else:
                new_dims.append(dim)
        
        new_data_vars[var_name] = (new_dims, var_data.values, var_data.attrs)
    
    # Create new dataset
    ds_new = xr.Dataset(new_data_vars, coords=new_coords, attrs=ds.attrs)
    
    # Ensure proper variable names for OpenDrift
    if 'u10' in ds_new.data_vars:
        ds_new = ds_new.rename({'u10': 'eastward_wind'})
        print("Renamed u10 → eastward_wind")
    
    if 'v10' in ds_new.data_vars:
        ds_new = ds_new.rename({'v10': 'northward_wind'}) 
        print("Renamed v10 → northward_wind")
    
    # Clean up coordinates - keep only essential ones
    essential_coords = ['time', 'latitude', 'longitude']
    coords_to_drop = [coord for coord in ds_new.coords if coord not in essential_coords]
    if coords_to_drop:
        print(f"Dropping coordinates: {coords_to_drop}")
        ds_new = ds_new.drop_vars(coords_to_drop)
    
    print(f"Final variables: {list(ds_new.data_vars)}")
    print(f"Final dimensions: {dict(ds_new.dims)}")
    print(f"Time range: {ds_new.time.values[0]} to {ds_new.time.values[-1]}")
    
    # Save to NetCDF
    ds_new.to_netcdf('wind.nc')
    
    # Close datasets
    ds.close()
    ds_new.close()
    
    print("✓ Successfully converted to wind.nc")
    print("Your wind data now has:")
    print(f"  - {len(valid_times)} time steps (3-hourly for 10 days)")
    print("  - eastward_wind and northward_wind variables")
    print("  - Ready for OpenDrift!")

if __name__ == "__main__":
    convert_ecmwf_wind()
