#!/usr/bin/env python3
"""
Inspect GRIB file structure to understand the data organization
"""
import xarray as xr
import cfgrib

def inspect_grib_file(filename='wind.grib2'):
    print(f"=== Inspecting {filename} ===\n")
    
    # Method 1: Simple open
    print("1. Simple open with cfgrib:")
    try:
        ds = xr.open_dataset(filename, engine='cfgrib')
        print(f"   Variables: {list(ds.data_vars)}")
        print(f"   Coordinates: {list(ds.coords)}")
        print(f"   Dimensions: {dict(ds.dims)}")
        if 'time' in ds.coords:
            print(f"   Time shape: {ds.time.shape}")
            print(f"   Time values: {ds.time.values}")
        if 'step' in ds.coords:
            print(f"   Step shape: {ds.step.shape}")
            print(f"   Step values: {ds.step.values}")
        ds.close()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Open multiple datasets
    print("2. Multiple datasets with cfgrib:")
    try:
        datasets = cfgrib.open_datasets(filename)
        print(f"   Found {len(datasets)} datasets:")
        for i, ds in enumerate(datasets):
            print(f"   Dataset {i}:")
            print(f"     Variables: {list(ds.data_vars)}")
            print(f"     Dimensions: {dict(ds.dims)}")
            if 'time' in ds.coords:
                print(f"     Time: {ds.time.values}")
            if 'step' in ds.coords:
                print(f"     Step: {ds.step.values}")
            print()
        
        # Close datasets
        for ds in datasets:
            ds.close()
            
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Method 3: Try with filtering
    print("3. Filtered for 10m wind:")
    try:
        ds = xr.open_dataset(filename, engine='cfgrib', 
                            backend_kwargs={'filter_by_keys': {
                                'typeOfLevel': 'heightAboveGround', 
                                'level': 10
                            }})
        print(f"   Variables: {list(ds.data_vars)}")
        print(f"   Dimensions: {dict(ds.dims)}")
        if 'time' in ds.coords:
            print(f"   Time: {ds.time.values}")
        ds.close()
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    inspect_grib_file()
