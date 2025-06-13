import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import os
import matplotlib.animation as animation
from matplotlib import cm

def create_currents_video(file_path, output_path='currents_video.mp4', depth_idx=0, fps=5):
    """Create a video of ocean currents with tides."""
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    # Determine total number of time steps
    time_steps = len(ds.time)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Find max speed for consistent color scale
    speed_max = 0
    for t in range(time_steps):
        u = ds['uo'].isel(time=t, depth=depth_idx)
        v = ds['vo'].isel(time=t, depth=depth_idx)
        u_tide = ds['utide'].isel(time=t, depth=depth_idx)
        v_tide = ds['vtide'].isel(time=t, depth=depth_idx)
        u_total = u + u_tide
        v_total = v + v_tide
        speed = np.sqrt(u_total**2 + v_total**2)
        speed_max = max(speed_max, float(speed.max()))
    
    # Initial plot for colorbar
    u = ds['uo'].isel(time=0, depth=depth_idx)
    v = ds['vo'].isel(time=0, depth=depth_idx)
    u_tide = ds['utide'].isel(time=0, depth=depth_idx)
    v_tide = ds['vtide'].isel(time=0, depth=depth_idx)
    u_total = u + u_tide
    v_total = v + v_tide
    speed = np.sqrt(u_total**2 + v_total**2)
    
    # Create initial plot for getting the colorbar
    speed_plot = ax.contourf(speed.longitude, speed.latitude, speed, 
                              cmap='viridis', transform=ccrs.PlateCarree(),
                              vmin=0, vmax=speed_max)
    plt.colorbar(speed_plot, ax=ax, label='Current Speed (m/s)')
    
    # Function to update each frame
    def update_frame(t):
        ax.clear()
        ax.coastlines(resolution='50m')
        
        # Extract variables for this time step
        u = ds['uo'].isel(time=t, depth=depth_idx)
        v = ds['vo'].isel(time=t, depth=depth_idx)
        u_tide = ds['utide'].isel(time=t, depth=depth_idx)
        v_tide = ds['vtide'].isel(time=t, depth=depth_idx)
        u_total = u + u_tide
        v_total = v + v_tide
        
        # Calculate speed
        speed = np.sqrt(u_total**2 + v_total**2)
        
        # Plot speed as contours with fixed scale
        speed_plot = speed.plot.contourf(ax=ax, cmap='viridis', transform=ccrs.PlateCarree(),
                                     vmin=0, vmax=speed_max, add_colorbar=False)
        
        # Plot arrows (subsampled)
        stride = 5
        ax.quiver(u_total.longitude[::stride], u_total.latitude[::stride],
                  u_total.values[::stride, ::stride], v_total.values[::stride, ::stride],
                  transform=ccrs.PlateCarree(), scale=20, width=0.002, color='white')
        
        # Add title with timestamp
        time_str = str(ds.time.values[t])[:19]
        ax.set_title(f'Currents with tides at {time_str}')
        
        return [ax]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=time_steps, blit=False)
    
    # Save as mp4
    ani.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=200)
    plt.close()
    
    # Close dataset
    ds.close()
    
    print(f"Currents video saved to {output_path}")

def create_waves_video(file_path, output_path='waves_video.mp4', fps=5):
    """Create a video of wave heights and directions."""
    # Open dataset
    ds = xr.open_dataset(file_path)
    
    # Determine total number of time steps
    time_steps = len(ds.time)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Determine max wave height for consistent color scale
    wave_height_max = float(ds['VHM0'].max())
    
    # Initial plot for colorbar
    wave_height = ds['VHM0'].isel(time=0)
    height_plot = ax.contourf(wave_height.longitude, wave_height.latitude, wave_height, 
                             cmap='viridis', transform=ccrs.PlateCarree(),
                             vmin=0, vmax=wave_height_max)
    plt.colorbar(height_plot, ax=ax, label='Significant Wave Height (m)')
    
    # Function to update each frame
    def update_frame(t):
        ax.clear()
        ax.coastlines(resolution='50m')
        
        # Extract wave height for this time step
        wave_height = ds['VHM0'].isel(time=t)
        
        # Plot wave height with fixed scale
        height_plot = wave_height.plot.contourf(ax=ax, cmap='viridis', transform=ccrs.PlateCarree(),
                                           vmin=0, vmax=wave_height_max, add_colorbar=False)
        
        # Add wave direction if available
        if 'VMDR' in ds:
            wave_dir = ds['VMDR'].isel(time=t)
            stride = 8
            # Convert direction to vector components
            u = -np.sin(np.deg2rad(wave_dir))
            v = -np.cos(np.deg2rad(wave_dir))
            ax.quiver(wave_dir.longitude[::stride], wave_dir.latitude[::stride],
                      u.values[::stride, ::stride], v.values[::stride, ::stride],
                      transform=ccrs.PlateCarree(), scale=30, width=0.002, color='white')
        
        # Add title with timestamp
        time_str = str(ds.time.values[t])[:19]
        ax.set_title(f'Significant Wave Height at {time_str}')
        
        return [ax]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=time_steps, blit=False)
    
    # Save as mp4
    ani.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=200)
    plt.close()
    
    # Close dataset
    ds.close()
    
    print(f"Waves video saved to {output_path}")

if __name__ == "__main__":
    create_currents_video('currents_tides.nc', 'currents_video.mp4')
    create_waves_video('waves.nc', 'waves_video.mp4')
