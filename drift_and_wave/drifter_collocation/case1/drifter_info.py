"""
Analyze the structure of your drifter NetCDF file to understand data organization
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_drifter_data(filename):
    """Analyze the drifter dataset structure"""
    
    print("="*80)
    print("DRIFTER DATA ANALYSIS")
    print("="*80)
    
    # Load dataset
    ds = xr.open_dataset(filename)
    
    print(f"\nFile: {filename}")
    print(f"Dimensions: {dict(ds.dims)}")
    
    print("\nVariables and their dimensions:")
    for var_name, var in ds.data_vars.items():
        print(f"  {var_name}: {var.dims} -> {var.shape}")
    
    print("\nCoordinate variables:")
    for coord_name, coord in ds.coords.items():
        print(f"  {coord_name}: {coord.dims} -> {coord.shape}")
    
    # Analyze GPS data (high frequency)
    print("\n" + "="*40)
    print("GPS DATA ANALYSIS (time dimension)")
    print("="*40)
    
    gps_time = ds['time']
    gps_lat = ds['lat'] 
    gps_lon = ds['lon']
    
    print(f"GPS time shape: {gps_time.shape}")
    print(f"GPS time range: {gps_time.min().values} to {gps_time.max().values}")
    
    # Count valid GPS points per trajectory
    for traj in range(ds.sizes['trajectory']):
        lat_traj = gps_lat.isel(trajectory=traj)
        lon_traj = gps_lon.isel(trajectory=traj)
        time_traj = gps_time.isel(trajectory=traj)
        
        valid_gps = (~np.isnan(lat_traj.values) & 
                    ~np.isnan(lon_traj.values) & 
                    ~pd.isnull(pd.to_datetime(time_traj.values)))
        
        valid_count = valid_gps.sum()
        total_count = len(lat_traj)
        
        print(f"  Trajectory {traj+1}: {valid_count}/{total_count} valid GPS points ({100*valid_count/total_count:.1f}%)")
        
        if valid_count > 0:
            first_valid = np.where(valid_gps)[0][0]
            last_valid = np.where(valid_gps)[0][-1]
            
            start_time = pd.to_datetime(time_traj.values[first_valid])
            end_time = pd.to_datetime(time_traj.values[last_valid])
            duration = end_time - start_time
            
            print(f"    Time span: {start_time} to {end_time} ({duration})")
            print(f"    Position: ({lat_traj.values[first_valid]:.3f}°N, {lon_traj.values[first_valid]:.3f}°E) to ({lat_traj.values[last_valid]:.3f}°N, {lon_traj.values[last_valid]:.3f}°E)")
    
    # Analyze wave data (lower frequency)
    if 'time_waves_imu' in ds.coords:
        print("\n" + "="*40)
        print("WAVE DATA ANALYSIS (time_waves_imu dimension)")
        print("="*40)
        
        wave_time = ds['time_waves_imu']
        print(f"Wave time shape: {wave_time.shape}")
        print(f"Wave time range: {wave_time.min().values} to {wave_time.max().values}")
        
        # Analyze wave variables
        wave_vars = ['Hs0', 'T02', 'T24', 'pHs0', 'pT02', 'pT24']
        available_wave_vars = [var for var in wave_vars if var in ds.data_vars]
        
        print(f"Available wave variables: {available_wave_vars}")
        
        for traj in range(ds.sizes['trajectory']):
            wave_time_traj = wave_time.isel(trajectory=traj)
            
            print(f"\n  Trajectory {traj+1} wave data:")
            
            for var_name in available_wave_vars:
                var_data = ds[var_name].isel(trajectory=traj)
                valid_wave = ~np.isnan(var_data.values)
                valid_count = valid_wave.sum()
                total_count = len(var_data)
                
                print(f"    {var_name}: {valid_count}/{total_count} valid points ({100*valid_count/total_count:.1f}%)")
                
                if valid_count > 0:
                    mean_val = np.nanmean(var_data.values)
                    std_val = np.nanstd(var_data.values)
                    min_val = np.nanmin(var_data.values)
                    max_val = np.nanmax(var_data.values)
                    
                    print(f"      Stats: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
    
    # Time alignment analysis
    print("\n" + "="*40)
    print("TIME ALIGNMENT ANALYSIS")
    print("="*40)
    
    if 'time_waves_imu' in ds.coords:
        for traj in range(min(3, ds.sizes['trajectory'])):  # Just check first 3 trajectories
            print(f"\nTrajectory {traj+1}:")
            
            gps_times = pd.to_datetime(ds['time'].isel(trajectory=traj).values)
            wave_times = pd.to_datetime(ds['time_waves_imu'].isel(trajectory=traj).values)
            
            # Remove NaT values
            gps_times_valid = gps_times[~pd.isnull(gps_times)]
            wave_times_valid = wave_times[~pd.isnull(wave_times)]
            
            if len(gps_times_valid) > 0 and len(wave_times_valid) > 0:
                gps_start = gps_times_valid.min()
                gps_end = gps_times_valid.max()
                wave_start = wave_times_valid.min()
                wave_end = wave_times_valid.max()
                
                print(f"  GPS time span: {gps_start} to {gps_end}")
                print(f"  Wave time span: {wave_start} to {wave_end}")
                
                # Calculate sampling intervals
                if len(gps_times_valid) > 1:
                    gps_interval = (gps_end - gps_start) / (len(gps_times_valid) - 1)
                    print(f"  Average GPS sampling interval: {gps_interval}")
                
                if len(wave_times_valid) > 1:
                    wave_interval = (wave_end - wave_start) / (len(wave_times_valid) - 1)
                    print(f"  Average wave sampling interval: {wave_interval}")
    
    # Create a quick visualization
    print("\n" + "="*40)
    print("CREATING SUMMARY PLOTS")
    print("="*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Data availability per trajectory
    ax1 = axes[0, 0]
    trajectories = []
    gps_counts = []
    wave_counts = []
    
    for traj in range(ds.sizes['trajectory']):
        trajectories.append(f"T{traj+1}")
        
        # GPS data count
        lat_traj = ds['lat'].isel(trajectory=traj)
        lon_traj = ds['lon'].isel(trajectory=traj)
        valid_gps = (~np.isnan(lat_traj.values) & ~np.isnan(lon_traj.values))
        gps_counts.append(valid_gps.sum())
        
        # Wave data count
        if 'Hs0' in ds.data_vars:
            hs_traj = ds['Hs0'].isel(trajectory=traj)
            valid_wave = ~np.isnan(hs_traj.values)
            wave_counts.append(valid_wave.sum())
        else:
            wave_counts.append(0)
    
    x = np.arange(len(trajectories))
    width = 0.35
    
    ax1.bar(x - width/2, gps_counts, width, label='GPS points', alpha=0.7)
    ax1.bar(x + width/2, wave_counts, width, label='Wave points', alpha=0.7)
    ax1.set_xlabel('Trajectory')
    ax1.set_ylabel('Number of valid points')
    ax1.set_title('Data Availability per Trajectory')
    ax1.set_xticks(x)
    ax1.set_xticklabels(trajectories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample Hs time series
    ax2 = axes[0, 1]
    if 'Hs0' in ds.data_vars and 'time_waves_imu' in ds.coords:
        for traj in range(min(3, ds.sizes['trajectory'])):  # Plot first 3 trajectories
            hs_data = ds['Hs0'].isel(trajectory=traj)
            wave_times = ds['time_waves_imu'].isel(trajectory=traj)
            
            valid = ~np.isnan(hs_data.values)
            if valid.any():
                times_pd = pd.to_datetime(wave_times.values[valid])
                hs_values = hs_data.values[valid]
                
                ax2.plot(times_pd, hs_values, 'o-', alpha=0.7, markersize=3, 
                        label=f'Trajectory {traj+1}')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Significant Wave Height (m)')
        ax2.set_title('Sample Hs Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Trajectory map
    ax3 = axes[1, 0]
    for traj in range(ds.sizes['trajectory']):
        lat_traj = ds['lat'].isel(trajectory=traj)
        lon_traj = ds['lon'].isel(trajectory=traj)
        
        valid = (~np.isnan(lat_traj.values) & ~np.isnan(lon_traj.values))
        if valid.any():
            ax3.plot(lon_traj.values[valid], lat_traj.values[valid], 
                    'o-', alpha=0.7, markersize=2, linewidth=1,
                    label=f'T{traj+1}')
    
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Drifter Trajectories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hs distribution
    ax4 = axes[1, 1]
    if 'Hs0' in ds.data_vars:
        all_hs = []
        for traj in range(ds.sizes['trajectory']):
            hs_data = ds['Hs0'].isel(trajectory=traj)
            valid_hs = hs_data.values[~np.isnan(hs_data.values)]
            all_hs.extend(valid_hs)
        
        if all_hs:
            ax4.hist(all_hs, bins=30, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Significant Wave Height (m)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Hs Distribution')
            ax4.grid(True, alpha=0.3)
            
            print(f"  Hs statistics: mean={np.mean(all_hs):.3f}m, std={np.std(all_hs):.3f}m")
            print(f"  Hs range: {np.min(all_hs):.3f}m to {np.max(all_hs):.3f}m")
    
    plt.tight_layout()
    plt.savefig('drifter_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved analysis plot as 'drifter_data_analysis.png'")
    
    ds.close()
    plt.show()

if __name__ == "__main__":
    analyze_drifter_data("2025-otc-omb.nc")
