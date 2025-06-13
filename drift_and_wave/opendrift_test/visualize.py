import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

def visualize_currents_with_tides(file_path, output_path='currents_tides.png', time_idx=0, depth_idx=0):
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    # Extract variables
    u = ds['uo'].isel(time=time_idx, depth=depth_idx)
    v = ds['vo'].isel(time=time_idx, depth=depth_idx)
    
    u_tide = ds['utide'].isel(time=time_idx, depth=depth_idx)
    v_tide = ds['vtide'].isel(time=time_idx, depth=depth_idx)
    
    u_total = u+u_tide
    v_total = v+v_tide

    # Calculate speed
    speed = np.sqrt(u_total**2 + v_total**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Basic map elements
    ax.coastlines(resolution='50m')
    
    # Plot speed as contours
    speed_plot = speed.plot.contourf(ax=ax, cmap='viridis', transform=ccrs.PlateCarree(),add_colorbar=False)
    plt.colorbar(speed_plot, ax=ax, label='Current Speed (m/s)')
    
    # Plot arrows (subsampled)
    stride = 5
    ax.quiver(u_total.longitude[::stride], u_total.latitude[::stride],
              u_total.values[::stride, ::stride], v_total.values[::stride, ::stride],
              transform=ccrs.PlateCarree(), scale=20, width=0.002, color='white')
    
    # Add title with timestamp
    time_str = str(ds.time.values[time_idx])[:19]
    plt.title(f'Currents with tides at {time_str}')
    
    # Save and close
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Close dataset
    ds.close()

def visualize_waves(file_path, output_path='waves.png', time_idx=0):
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    # Extract wave height
    wave_height = ds['VHM0'].isel(time=time_idx)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Basic map elements
    ax.coastlines(resolution='50m')
    
    # Plot wave height
    height_plot = wave_height.plot.contourf(ax=ax, cmap='viridis', transform=ccrs.PlateCarree(),add_colorbar=False)
    plt.colorbar(height_plot, ax=ax, label='Significant Wave Height (m)')
    
    # Add wave direction if available
    if 'VMDR' in ds:
        wave_dir = ds['VMDR'].isel(time=time_idx)
        stride = 8
        # Convert direction to vector components
        u = -np.sin(np.deg2rad(wave_dir))
        v = -np.cos(np.deg2rad(wave_dir))
        ax.quiver(wave_dir.longitude[::stride], wave_dir.latitude[::stride],
                  u.values[::stride, ::stride], v.values[::stride, ::stride],
                  transform=ccrs.PlateCarree(), scale=30, width=0.002, color='white')
    
    # Add title
    time_str = str(ds.time.values[time_idx])[:19]
    plt.title(f'Hs at {time_str}')
    
    # Save and close
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Close dataset
    ds.close()

if __name__ == "__main__":
    visualize_currents_with_tides('currents_tides.nc', 'currents_tides.png',72)
    visualize_waves('waves.nc', 'waves.png',24)
