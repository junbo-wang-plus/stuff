import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

def haversine_km(lon1, lat1, lon2, lat2):
    """Fast haversine distance in kilometers"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return 6371 * c  # Earth radius in km

def collocate_satellite_csv_and_plot(drifter_file, satellite_csv_files, satellite_names=None, 
                                    time_window_hours=3, radius_km=50):
    """Collocation with CSV satellite products using time window and spatial radius"""
    
    # Default satellite names if not provided
    if satellite_names is None:
        satellite_names = [f'Satellite_{i+1}' for i in range(len(satellite_csv_files))]
    
    # Load drifter dataset
    drifter_ds = xr.open_dataset(drifter_file)
    
    # Load all satellite CSV files
    satellite_dataframes = []
    for sat_file in satellite_csv_files:
        try:
            sat_df = pd.read_csv(sat_file)
            sat_df['time'] = pd.to_datetime(sat_df['time']).dt.tz_localize(None)
            satellite_dataframes.append(sat_df)
            print(f"Loaded {sat_file}: {len(sat_df)} measurements")
        except Exception as e:
            satellite_dataframes.append(None)
    
    # Get drifter data
    drifter_lats = drifter_ds['lat'].values
    drifter_lons = drifter_ds['lon'].values
    drifter_times = drifter_ds['time'].values
    wave_measurement_times = drifter_ds['time_waves_imu'].values
    n_trajectories, n_wave_times = wave_measurement_times.shape
    
    # Initialize lists to store collocated data
    satellite_collocation = {}
    for sat_name in satellite_names:
        satellite_collocation[sat_name] = {
            'drifter_id': [],
            'time_index': [],
            'wave_height': []
        }
    
    # Collocation loop
    print(f"Starting collocation for {n_wave_times} time steps...")
    for t_idx in range(n_wave_times):
        if t_idx % 50 == 0:
            print(f"Processing step {t_idx}/{n_wave_times}")
            
        current_wave_times = wave_measurement_times[:, t_idx]
        if np.all(pd.isna(current_wave_times)):
            continue
        
        # Process each drifter
        for traj_idx in range(n_trajectories):
            wave_time = current_wave_times[traj_idx]
            if pd.isna(wave_time):
                continue
            
            # Find closest position time
            traj_position_times = drifter_times[traj_idx, :]
            valid_pos_times = ~pd.isna(traj_position_times)
            if not valid_pos_times.any():
                continue
                
            time_diffs = np.abs(pd.to_datetime(traj_position_times[valid_pos_times]) - pd.to_datetime(wave_time))
            closest_pos_idx = np.where(valid_pos_times)[0][time_diffs.argmin()]
            
            lat_pos = drifter_lats[traj_idx, closest_pos_idx]
            lon_pos = drifter_lons[traj_idx, closest_pos_idx]
            
            if (np.isnan(lat_pos) or np.isnan(lon_pos) or 
                time_diffs.min().total_seconds() / 3600 > 1.0):
                continue
            
            drifter_time = pd.to_datetime(wave_time)
            drifter_pos = (lat_pos, lon_pos)
            
            # Check each satellite for nearby measurements
            for sat_idx, (sat_df, sat_name) in enumerate(zip(satellite_dataframes, satellite_names)):
                if sat_df is None:
                    continue
                
                # Filter by time window
                time_lower = drifter_time - pd.Timedelta(hours=time_window_hours)
                time_upper = drifter_time + pd.Timedelta(hours=time_window_hours)
                time_mask = (sat_df['time'] >= time_lower) & (sat_df['time'] <= time_upper)
                nearby_times = sat_df[time_mask]
                
                if len(nearby_times) == 0:
                    continue
                
                # Filter by spatial radius
                for _, sat_row in nearby_times.iterrows():
                    distance = haversine_km(lon_pos, lat_pos, sat_row['longitude'], sat_row['latitude'])
                    
                    if distance <= radius_km:
                        satellite_collocation[sat_name]['drifter_id'].append(traj_idx)
                        satellite_collocation[sat_name]['time_index'].append(t_idx)
                        satellite_collocation[sat_name]['wave_height'].append(sat_row['value'])
    
    print("Collocation complete!")
    
    # Create timeseries plot
    n_drifters = n_trajectories
    n_cols = 2
    n_rows = (n_drifters + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get observed data
    hs_obs = drifter_ds['Hs0'].values
    
    # Define colors and markers for satellites
    satellite_styles = [
        {'color': 'red', 'marker': 'o', 'size': 40},
        {'color': 'green', 'marker': 's', 'size': 35},
        {'color': 'orange', 'marker': '^', 'size': 40},
        {'color': 'purple', 'marker': 'D', 'size': 30},
        {'color': 'brown', 'marker': 'v', 'size': 40},
        {'color': 'pink', 'marker': 'p', 'size': 40},
        {'color': 'cyan', 'marker': 'h', 'size': 40},
        {'color': 'gray', 'marker': '*', 'size': 50},
        {'color': 'lime', 'marker': 'X', 'size': 40},
        {'color': 'navy', 'marker': '+', 'size': 60},
    ]
    
    for drifter_idx in range(n_drifters):
        row = drifter_idx // n_cols
        col = drifter_idx % n_cols
        ax = axes[row, col]
        
        obs_data = hs_obs[drifter_idx, :]
        time_indices = np.arange(len(obs_data))
        
        # Plot observed data
        valid_obs = ~np.isnan(obs_data)
        if valid_obs.any():
            ax.plot(time_indices[valid_obs], obs_data[valid_obs], 
                   'o-', color='blue', linewidth=2, markersize=3, 
                   alpha=0.7, label='Drifter obs')
        
        # Plot each satellite as scatter points
        for sat_idx, sat_name in enumerate(satellite_names):
            sat_data = satellite_collocation[sat_name]
            drifter_mask = np.array(sat_data['drifter_id']) == drifter_idx
            
            if np.any(drifter_mask):
                sat_times = np.array(sat_data['time_index'])[drifter_mask]
                sat_values = np.array(sat_data['wave_height'])[drifter_mask]
                
                if len(sat_values) > 0:
                    style = satellite_styles[sat_idx % len(satellite_styles)]
                    ax.scatter(sat_times, sat_values,
                              c=style['color'], marker=style['marker'], 
                              s=style['size'], alpha=0.8,
                              label=f"{sat_name} (n={len(sat_values)})", 
                              edgecolors='white', linewidths=0.5)
        
        ax.set_title(f'Drifter {drifter_idx+1}')
        ax.set_xlabel('Time index')
        ax.set_ylabel('Wave height (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for drifter_idx in range(n_drifters, n_rows * n_cols):
        row = drifter_idx // n_cols
        col = drifter_idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('drifter_satellite_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Close datasets
    drifter_ds.close()

def plot_satellite_tracks_and_drifters(drifter_file, satellite_csv_files, satellite_names=None):
    """Plot satellite tracks and drifter trajectories on a map"""
    
    print("Creating satellite tracks and drifter trajectories map...")
    
    # Default satellite names if not provided
    if satellite_names is None:
        satellite_names = [f'Satellite_{i+1}' for i in range(len(satellite_csv_files))]
    
    # Load drifter dataset
    drifter_ds = xr.open_dataset(drifter_file)
    
    # Load all satellite CSV files
    satellite_dataframes = []
    for sat_file in satellite_csv_files:
        try:
            sat_df = pd.read_csv(sat_file)
            satellite_dataframes.append(sat_df)
        except Exception as e:
            satellite_dataframes.append(None)
    
    # Get drifter trajectories
    drifter_lats = drifter_ds['lat'].values
    drifter_lons = drifter_ds['lon'].values
    n_trajectories = drifter_lats.shape[0]
    
    # Calculate map extent
    all_lats = []
    all_lons = []
    
    # Add drifter positions
    valid_drifter_lats = drifter_lats[~np.isnan(drifter_lats)]
    valid_drifter_lons = drifter_lons[~np.isnan(drifter_lons)]
    all_lats.extend(valid_drifter_lats)
    all_lons.extend(valid_drifter_lons)
    
    # Add satellite positions
    for sat_df in satellite_dataframes:
        if sat_df is not None:
            all_lats.extend(sat_df['latitude'].values)
            all_lons.extend(sat_df['longitude'].values)
    
    # Calculate extent with padding
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)
    padding = 0.1
    
    extent = [
        min(all_lons) - padding * lon_range,
        max(all_lons) + padding * lon_range,
        min(all_lats) - padding * lat_range,
        max(all_lats) + padding * lat_range
    ]
    
    # Create map
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, color='gray', linewidth=0.5)
    
    # Define colors for satellites
    satellite_colors = [
        'red', 'green', 'orange', 'purple', 'brown',
        'pink', 'cyan', 'gray', 'lime', 'navy'
    ]
    
    # Plot satellite tracks (subsample for performance)
    for sat_idx, (sat_df, sat_name) in enumerate(zip(satellite_dataframes, satellite_names)):
        if sat_df is not None and len(sat_df) > 0:
            # Subsample large datasets for visualization
            if len(sat_df) > 5000:
                sat_df_plot = sat_df.sample(n=5000, random_state=42)
            else:
                sat_df_plot = sat_df
                
            color = satellite_colors[sat_idx % len(satellite_colors)]
            ax.scatter(sat_df_plot['longitude'], sat_df_plot['latitude'], 
                      c=color, s=1, alpha=0.6, 
                      label=f'{sat_name} ({len(sat_df)} pts)',
                      transform=ccrs.PlateCarree())
    
    # Plot drifter trajectories
    drifter_trajectory_colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'purple',
                                'goldenrod', 'darkcyan', 'darkmagenta', 'darkslategray', 'maroon']
    
    for traj_idx in range(n_trajectories):
        lat_traj = drifter_lats[traj_idx, :]
        lon_traj = drifter_lons[traj_idx, :]
        
        valid = ~(np.isnan(lat_traj) | np.isnan(lon_traj))
        if valid.any():
            color = drifter_trajectory_colors[traj_idx % len(drifter_trajectory_colors)]
            
            # Plot trajectory line
            ax.plot(lon_traj[valid], lat_traj[valid], 
                   color=color, linewidth=3, alpha=0.9,
                   label=f'Drifter {traj_idx+1}',
                   transform=ccrs.PlateCarree())
            
            # Mark start and end points
            lat_vals = lat_traj[valid]
            lon_vals = lon_traj[valid]
            ax.scatter(lon_vals[0], lat_vals[0], 
                      marker='o', s=100, color='white', 
                      edgecolor=color, linewidth=3,
                      transform=ccrs.PlateCarree(), zorder=10)
            ax.scatter(lon_vals[-1], lat_vals[-1], 
                      marker='s', s=100, color=color, 
                      edgecolor='white', linewidth=2,
                      transform=ccrs.PlateCarree(), zorder=10)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, color='white', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title('Satellite Tracks and Drifter Trajectories', fontsize=16, fontweight='bold')
    
    # Add legend (split into two columns to fit)
    handles, labels = ax.get_legend_handles_labels()
    
    # Split legend into satellites and drifters
    sat_handles = handles[:len(satellite_names)]
    sat_labels = labels[:len(satellite_names)]
    drifter_handles = handles[len(satellite_names):]
    drifter_labels = labels[len(satellite_names):]
    
    # Create two legends
    if sat_handles:
        sat_legend = ax.legend(sat_handles, sat_labels, loc='upper left', 
                              bbox_to_anchor=(0.02, 0.98), fontsize=9, 
                              title='Satellite Tracks', title_fontsize=10)
        ax.add_artist(sat_legend)
    
    if drifter_handles:
        ax.legend(drifter_handles, drifter_labels, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=9,
                 title='Drifter Trajectories', title_fontsize=10)
    
    plt.savefig('satellite_tracks_and_drifters.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Close dataset
    drifter_ds.close()
    
    print(f"Map saved as: satellite_tracks_and_drifters.png")

# Run with satellite CSV files
if __name__ == "__main__":
    satellite_csv_files = [
        "cfo.csv", "c2.csv", "h2b.csv", "h2c.csv", "j3.csv",
        "al.csv", "s3a.csv", "s3b.csv", "s6a.csv", "swon.csv"
    ]
    
    satellite_names = [
        "CFOSAT", "CryoSat-2", "HaiYang-2B", "HaiYang-2C", "JASON-3",
        "AltiKa", "Sentinel-3A", "Sentinel-3B", "Sentinel-6A", "SWOT"
    ]
    
    # Create timeseries comparison
    collocate_satellite_csv_and_plot(
        drifter_file="2025-otc-omb.nc",
        satellite_csv_files=satellite_csv_files,
        satellite_names=satellite_names,
        time_window_hours=3,
        radius_km=50
    )
    
    # Create satellite tracks and drifter trajectories map
    plot_satellite_tracks_and_drifters(
        drifter_file="2025-otc-omb.nc",
        satellite_csv_files=satellite_csv_files,
        satellite_names=satellite_names
    )
