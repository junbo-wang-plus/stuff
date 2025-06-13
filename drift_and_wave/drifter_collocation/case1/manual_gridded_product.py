import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def collocate_and_plot(drifter_file, wave_file):
    """Minimal collocation and timeseries plotting"""
    
    # Load datasets
    drifter_ds = xr.open_dataset(drifter_file)
    wave_ds = xr.open_dataset(wave_file)
    
    # Get drifter data
    drifter_lats = drifter_ds['lat'].values
    drifter_lons = drifter_ds['lon'].values
    drifter_times = drifter_ds['time'].values
    wave_measurement_times = drifter_ds['time_waves_imu'].values
    n_trajectories, n_wave_times = wave_measurement_times.shape
    
    # Initialize forecast array
    hs_forecast = np.full((n_trajectories, n_wave_times), np.nan)
    
    # Collocation loop
    for t_idx in range(n_wave_times):
        current_wave_times = wave_measurement_times[:, t_idx]
        if np.all(pd.isna(current_wave_times)):
            continue
        
        valid_drifters = []
        drifter_positions = []
        
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
            
            if (not np.isnan(lat_pos) and not np.isnan(lon_pos) and 
                time_diffs.min().total_seconds() / 3600 < 1.0):
                valid_drifters.append(traj_idx)
                drifter_positions.append((lat_pos, lon_pos, wave_time))
        
        if not valid_drifters:
            continue
            
        # Find closest wave forecast time
        reference_time = drifter_positions[0][2]
        drifter_time = pd.to_datetime(reference_time).to_numpy()
        wave_times = pd.to_datetime(wave_ds.time.values)
        wave_time_diff = np.abs(wave_times - drifter_time)
        wave_time_idx = wave_time_diff.argmin()
        
        if wave_time_diff.min() / np.timedelta64(1, 'h') > 3.0:
            continue
        
        # Interpolate VHM0 to drifter positions
        wave_slice = wave_ds.isel(time=wave_time_idx)
        hs_data = wave_slice['VHM0'].values
        
        if np.any(~np.isnan(hs_data)):
            lats = wave_slice.latitude.values
            lons = wave_slice.longitude.values
            interpolator = RegularGridInterpolator(
                (lats, lons), hs_data, 
                method='linear', bounds_error=False, fill_value=np.nan
            )
            
            for i, traj_idx in enumerate(valid_drifters):
                lat_pos, lon_pos, _ = drifter_positions[i]
                drifter_pos = np.array([[lat_pos, lon_pos]])
                hs_forecast[traj_idx, t_idx] = interpolator(drifter_pos)[0]
    
    # Create timeseries plot
    n_drifters = n_trajectories
    n_cols = 2
    n_rows = (n_drifters + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get observed data
    hs_obs = drifter_ds['Hs0'].values
    
    for drifter_idx in range(n_drifters):
        row = drifter_idx // n_cols
        col = drifter_idx % n_cols
        ax = axes[row, col]
        
        obs_data = hs_obs[drifter_idx, :]
        fcst_data = hs_forecast[drifter_idx, :]
        
        valid = ~(np.isnan(obs_data) | np.isnan(fcst_data))
        
        if valid.any():
            time_indices = np.arange(len(obs_data))
            valid_indices = time_indices[valid]
            
            # Plot
            ax.plot(valid_indices, obs_data[valid], 'o-', color='blue', 
                   linewidth=2, markersize=4, label='Observed')
            ax.plot(valid_indices, fcst_data[valid], 's--', color='red', 
                   linewidth=1.5, markersize=3, label='Forecast')
            
            # Stats
            bias = np.mean(fcst_data[valid] - obs_data[valid])
            rmse = np.sqrt(np.mean((fcst_data[valid] - obs_data[valid])**2))
            
            ax.set_title(f'Drifter {drifter_idx+1}\nBias: {bias:.3f}m, RMSE: {rmse:.3f}m')
        else:
            ax.text(0.5, 0.5, f'Drifter {drifter_idx+1}\nNo valid data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Time index')
        ax.set_ylabel('Wave height (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for drifter_idx in range(n_drifters, n_rows * n_cols):
        row = drifter_idx // n_cols
        col = drifter_idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('drifter_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Close datasets
    drifter_ds.close()
    wave_ds.close()

# Run
if __name__ == "__main__":
    collocate_and_plot("2025-otc-omb.nc", "waves.nc")
