#!/usr/bin/env python3

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation

def visualize_temperature_at_depth(file_path, output_path='temperature.png', time_idx=0, depth_idx=0, depth_value=None):
    
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    temp_var = 'thetao'
    if 'depth' in ds.dims:
        if depth_value is not None:
            # Find closest depth to specified value
            depth_diff = np.abs(ds.depth.values - depth_value)
            depth_idx = np.argmin(depth_diff)
            actual_depth = ds.depth.values[depth_idx]
            print(f"depth: {actual_depth}m")
        else:
            actual_depth = ds.depth.values[depth_idx]
            print(f"depth: {actual_depth}m")
        
        # Extract temperature at specified depth
        temperature = ds[temp_var].isel(time=time_idx, depth=depth_idx)
    else:
        # 2D data (surface only)
        temperature = ds[temp_var].isel(time=time_idx)
        actual_depth = 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    
    # Plot temperature
    temp_plot = temperature.plot.contourf(
        ax=ax, 
        cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (warm to cool)
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        levels=20
    )
    
    # Add colorbar
    cbar = plt.colorbar(temp_plot, ax=ax, orientation='horizontal', 
                       pad=0.08, shrink=0.8, aspect=30)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title with timestamp and depth
    time_str = str(ds.time.values[time_idx])[:19]
    if actual_depth == 0:
        depth_str = "Surface"
    else:
        depth_str = f"{actual_depth}m depth"
    
    plt.title(f'Sea Water Temperature - {depth_str}\n{time_str}', fontsize=14, pad=20)
    
    # Set extent to data bounds with some padding
    lon_min, lon_max = float(temperature.longitude.min()), float(temperature.longitude.max())
    lat_min, lat_max = float(temperature.latitude.min()), float(temperature.latitude.max())
    padding = 2  # degrees
    ax.set_extent([lon_min-padding, lon_max+padding, lat_min-padding, lat_max+padding], 
                  ccrs.PlateCarree())
    
    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Close dataset
    ds.close()
    

def plot_temperature_depth_profile(file_path, output_path='temp_profile.png', 
                                 lon_target=-30.0, lat_target=50.0, time_idx=0):
    
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    # Get temperature variable
    temp_var = None
    for var_name in ['thetao', 'temperature', 'temp']:
        if var_name in ds.data_vars:
            temp_var = var_name
            break
    
    lon_diff = np.abs(ds.longitude.values - lon_target)
    lat_diff = np.abs(ds.latitude.values - lat_target)
    
    lon_idx = np.argmin(lon_diff)
    lat_idx = np.argmin(lat_diff)
    
    actual_lon = ds.longitude.values[lon_idx]
    actual_lat = ds.latitude.values[lat_idx]
    
    print(f"Nearest point: {actual_lon:.2f}°E, {actual_lat:.2f}°N")
    
    # Extract temperature profile at this location
    temp_profile = ds[temp_var].isel(time=time_idx, longitude=lon_idx, latitude=lat_idx)
    
    # Get depth values (convert to positive values for plotting)
    depths = ds.depth.values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Plot temperature vs depth
    ax.plot(temp_profile.values, depths, 'bo-', linewidth=2, markersize=6)
    
    # Invert y-axis so surface (0m) is at top
    ax.invert_yaxis()
    
    # Labels and formatting
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Title with location and time
    time_str = str(ds.time.values[time_idx])[:19]
    ax.set_title(f'Temperature Depth Profile\n'
                f'Location: {actual_lon:.2f}°E, {actual_lat:.2f}°N\n'
                f'Time: {time_str}', fontsize=12)
    
    # Add temperature range info
    temp_min, temp_max = float(temp_profile.min()), float(temp_profile.max())
    ax.text(0.02, 0.98, f'Range: {temp_min:.1f}°C to {temp_max:.1f}°C', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Close dataset
    ds.close()
    

def create_temperature_video(file_path, output_path='temperature_video.mp4', depth_value=0.0, fps=3):
    """
    Create a video of temperature evolution over time at specified depth.
    
    Parameters:
    -----------
    file_path : str
        Path to temperature NetCDF file
    output_path : str
        Output filename for the video
    depth_value : float
        Depth in meters (0.0 for surface, 50.0 for 50m depth, etc.)
    fps : int
        Frames per second for the video
    """
    
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    temp_var = 'thetao'
    
    # Determine depth handling
    if 'depth' in ds.dims:
        # Find closest depth to specified value
        depth_diff = np.abs(ds.depth.values - depth_value)
        depth_idx = np.argmin(depth_diff)
        actual_depth = ds.depth.values[depth_idx]
        print(f"Creating video at depth: {actual_depth}m")
        
        # Get temperature data for all time steps at this depth
        temp_data = ds[temp_var].isel(depth=depth_idx)
    else:
        # 2D data (surface only)
        temp_data = ds[temp_var]
        actual_depth = 0
        print("Creating surface temperature video")
    
    # Get time steps
    time_steps = len(ds.time)
    print(f"Processing {time_steps} time steps...")
    
    # Find temperature range for consistent color scale
    temp_min = float(temp_data.min())
    temp_max = float(temp_data.max())
    print(f"Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set extent based on data bounds
    lon_min, lon_max = float(temp_data.longitude.min()), float(temp_data.longitude.max())
    lat_min, lat_max = float(temp_data.latitude.min()), float(temp_data.latitude.max())
    padding = 2  # degrees
    ax.set_extent([lon_min-padding, lon_max+padding, lat_min-padding, lat_max+padding], 
                  ccrs.PlateCarree())
    
    def update_frame(t):
        ax.clear()
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        
        # Get temperature for this time step
        temperature = temp_data.isel(time=t)
        
        # Plot temperature with fixed color scale
        temp_plot = temperature.plot.contourf(
            ax=ax,
            cmap='RdYlBu_r',
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            levels=20,
            vmin=temp_min,
            vmax=temp_max
        )
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set extent again (sometimes gets reset)
        ax.set_extent([lon_min-padding, lon_max+padding, lat_min-padding, lat_max+padding], 
                      ccrs.PlateCarree())
        
        # Add title with timestamp and depth
        time_str = str(ds.time.values[t])[:19]
        if actual_depth == 0:
            depth_str = "Surface"
        else:
            depth_str = f"{actual_depth}m depth"
        
        ax.set_title(f'Sea Water Temperature - {depth_str}\n{time_str}', fontsize=14)
        
        return [ax]
    
    # Create colorbar (outside animation loop)
    # Use a dummy plot to get colorbar
    dummy_temp = temp_data.isel(time=0)
    dummy_plot = dummy_temp.plot.contourf(
        ax=ax, cmap='RdYlBu_r', transform=ccrs.PlateCarree(),
        add_colorbar=False, levels=20, vmin=temp_min, vmax=temp_max
    )
    cbar = plt.colorbar(dummy_plot, ax=ax, orientation='horizontal', 
                       pad=0.08, shrink=0.8, aspect=30)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Create animation
    print("Creating animation...")
    ani = animation.FuncAnimation(fig, update_frame, frames=time_steps, blit=False)
    
    # Save as mp4
    print(f"Saving video: {output_path}")
    ani.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=150)
    plt.close()
    
    # Close dataset
    ds.close()
    
    print(f"✓ Temperature video saved: {output_path}")


def main():
    
    # File path - adjust as needed
    temp_file = 'sst_forecast.nc'  # or 'temperature_forecast.nc'
    
    print("Temperature Data Visualization")
    print("=" * 50)
    
    # 1. Surface temperature plot
    print("\n1. Creating surface temperature plot...")
    visualize_temperature_at_depth(
        file_path=temp_file,
        output_path='temperature_surface.png',
        time_idx=0,
        depth_value=0.0  # Surface
    )
    
    # 2. Temperature at 50m depth
    print("\n2. Creating temperature plot at 50m depth...")
    visualize_temperature_at_depth(
        file_path=temp_file,
        output_path='temperature_50m.png',
        time_idx=0,
        depth_value=50.0  # 50m depth
    )
    
    # 3. Temperature depth profile at hardcoded location
    print("\n3. Creating temperature depth profile...")
    plot_temperature_depth_profile(
        file_path=temp_file,
        output_path='temperature_profile.png',
        lon_target=-30.0,  
        lat_target=50.0,   
        time_idx=0
    )
    
    # 4. Surface temperature video
    print("\n4. Creating surface temperature video...")
    create_temperature_video(
        file_path=temp_file,
        output_path='temperature_surface_video.mp4',
        depth_value=0.0,  # Surface
        fps=3
    )
    
    
      
if __name__ == "__main__":
    main()
