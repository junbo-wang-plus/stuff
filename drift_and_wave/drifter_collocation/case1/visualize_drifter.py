import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def quick_drifter_check(filename):
    """Quick check of drifter data ranges and OpenDrift-style plotting"""
    
    # Load dataset
    ds = xr.open_dataset(filename)
    
    print("=== QUICK DATA SUMMARY ===")
    print(f"Trajectories: {ds.sizes['trajectory']}")
    print(f"Time points per trajectory: {ds.sizes.get('time', 'N/A')}")
    
    # Time range
    time = ds['time']
    time_flat = time.values.flatten()
    time_valid = time_flat[~np.isnat(time_flat)]
    
    if len(time_valid) > 0:
        print(f"Time range: {time_valid.min()} to {time_valid.max()}")
    
    # Spatial range
    lat = ds['lat'].values.flatten()
    lon = ds['lon'].values.flatten()
    lat_valid = lat[~np.isnan(lat)]
    lon_valid = lon[~np.isnan(lon)]
    
    print(f"Latitude: {lat_valid.min():.3f}° to {lat_valid.max():.3f}°")
    print(f"Longitude: {lon_valid.min():.3f}° to {lon_valid.max():.3f}°")
    
    # ========== TRAJECTORY PLOT ==========
    fig1 = plt.figure(figsize=(15, 12))
    
    # Calculate plot extent with some padding
    lat_range = lat_valid.max() - lat_valid.min()
    lon_range = lon_valid.max() - lon_valid.min()
    padding = 0.1
    
    extent = [
        lon_valid.min() - padding * lon_range,
        lon_valid.max() + padding * lon_range,
        lat_valid.min() - padding * lat_range,
        lat_valid.max() + padding * lat_range
    ]
    
    # Main trajectory plot (OpenDrift style)
    ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add clean background
    ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.8)
    ax1.add_feature(cfeature.COASTLINE, color='black', linewidth=0.8)
    ax1.add_feature(cfeature.BORDERS, color='gray', linewidth=0.5)
    
    # Plot trajectories in OpenDrift style
    trajectory_colors = ['red', 'orange', 'yellow', 'lime', 'cyan', 
                        'magenta', 'pink', 'lightblue', 'lightgreen', 'gold']
    
    for i in range(ds.sizes['trajectory']):
        lat_traj = ds['lat'].isel(trajectory=i)
        lon_traj = ds['lon'].isel(trajectory=i)
        
        valid = ~(np.isnan(lat_traj) | np.isnan(lon_traj))
        if valid.any():
            color = trajectory_colors[i % len(trajectory_colors)]
            
            # Plot trajectory line
            ax1.plot(lon_traj[valid], lat_traj[valid], 
                    color=color, linewidth=2, alpha=0.8,
                    transform=ccrs.PlateCarree())
            
            # Add direction arrows along trajectory
            lat_vals = lat_traj[valid].values
            lon_vals = lon_traj[valid].values
            
            # Add arrows every N points
            arrow_interval = max(1, len(lat_vals) // 10)
            for j in range(0, len(lat_vals) - 1, arrow_interval):
                if j + 1 < len(lat_vals):
                    dx = lon_vals[j+1] - lon_vals[j]
                    dy = lat_vals[j+1] - lat_vals[j]
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Only if there's movement
                        ax1.arrow(lon_vals[j], lat_vals[j], dx*0.3, dy*0.3,
                                head_width=0.02, head_length=0.02, 
                                fc=color, ec=color, alpha=0.7,
                                transform=ccrs.PlateCarree())
            
            # Mark start and end points
            ax1.scatter(lon_vals[0], lat_vals[0], 
                       marker='o', s=80, color='white', 
                       edgecolor=color, linewidth=2,
                       transform=ccrs.PlateCarree(), zorder=10)
            ax1.scatter(lon_vals[-1], lat_vals[-1], 
                       marker='s', s=80, color=color, 
                       edgecolor='white', linewidth=2,
                       transform=ccrs.PlateCarree(), zorder=10)
    
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, alpha=0.5, color='white', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Title with time information
    if len(time_valid) > 0:
        start_time = str(time_valid.min())[:19]
        end_time = str(time_valid.max())[:19] 
        ax1.set_title(f'Drifter Trajectories\n{start_time} to {end_time} UTC ({len(time_valid)} steps)', 
                     fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Drifter Trajectories', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', label='Trajectory paths'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                   markeredgecolor='black', markersize=8, label='Start points'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markeredgecolor='white', markersize=8, label='End points')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save trajectory plot
    trajectory_filename = filename.replace('.nc', '_trajectories.png')
    fig1.savefig(trajectory_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"Trajectory plot saved as: {trajectory_filename}")
    
    # ========== WAVE HEIGHT PLOT ==========
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_facecolor('#f8f9fa')
    
    if 'Hs0' in ds.data_vars:
        for i in range(min(10, ds.sizes['trajectory'])):
            hs = ds['Hs0'].isel(trajectory=i)
            valid = ~np.isnan(hs)
            if valid.any():
                color = trajectory_colors[i % len(trajectory_colors)]
                time_indices = np.arange(len(hs[valid]))
                ax2.plot(time_indices, hs[valid], color=color, linewidth=2, alpha=0.8, 
                        label=f'Drifter {i+1}')
        
        ax2.set_xlabel('Time index', fontsize=12)
        ax2.set_ylabel('Significant Wave Height (m)', fontsize=12)
        ax2.set_title('Wave Height Time Series', fontsize=14, fontweight='bold')
        if ds.sizes['trajectory'] <= 10:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Save wave height plot
        waveheight_filename = filename.replace('.nc', '_wave_heights.png')
        fig2.savefig(waveheight_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"Wave height plot saved as: {waveheight_filename}")
    else:
        ax2.text(0.5, 0.5, 'No Hs0 data available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='gray')
        ax2.set_title('Wave Height Data', fontsize=14, fontweight='bold')
        
        # Save empty wave height plot
        waveheight_filename = filename.replace('.nc', '_wave_heights.png')
        fig2.savefig(waveheight_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"Wave height plot (no data) saved as: {waveheight_filename}")
    
    # Print trajectory details and data coverage info
    print(f"\n=== TRAJECTORY DETAILS ===")
    total_points = 0
    valid_points = 0
    
    for i in range(ds.sizes['trajectory']):
        lat_traj = ds['lat'].isel(trajectory=i).values
        lon_traj = ds['lon'].isel(trajectory=i).values
        time_traj = ds['time'].isel(trajectory=i).values
        
        valid = ~(np.isnan(lat_traj) | np.isnan(lon_traj))
        traj_valid = valid.sum()
        traj_total = len(lat_traj)
        
        total_points += traj_total
        valid_points += traj_valid
        
        # Find first valid position and time
        if traj_valid > 0:
            first_valid_idx = np.where(valid)[0][0]
            start_lat = lat_traj[first_valid_idx]
            start_lon = lon_traj[first_valid_idx]
            start_time = time_traj[first_valid_idx]
            
            # Format time nicely
            if not np.isnat(start_time):
                start_time_str = str(start_time)[:19]
            else:
                start_time_str = "N/A"
            
            print(f"Drifter {i+1}:")
            print(f"  Start: {start_lat:.3f}°N, {start_lon:.3f}°E at {start_time_str}")
            print(f"  Data: {traj_valid}/{traj_total} valid positions ({100*traj_valid/traj_total:.1f}%)")
        else:
            print(f"Drifter {i+1}: No valid data")
    
    print(f"\nOverall: {valid_points}/{total_points} valid positions ({100*valid_points/total_points:.1f}%)")
    
    ds.close()

# Usage
if __name__ == "__main__":
    filename = "2025-otc-omb.nc"  # Replace with your file path
    quick_drifter_check(filename)
