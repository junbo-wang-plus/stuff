#!/usr/bin/env python3

from ecmwf.opendata import Client
from datetime import datetime, timedelta
import sys

def download_ecmwf_wind_forecast(forecast_days=15, output_file=None):
    
    client = Client(source="ecmwf")  # or "azure" for faster downloads
    
    
    if forecast_days <= 6:  # Up to 144 hours
        max_hours = min(forecast_days * 24, 144)
        steps = list(range(0, max_hours + 1, 3))
    else:  # Extended range (>6 days)
        # 0-144h by 3h: [0, 3, 6, 9, ..., 144]
        steps = list(range(0, 145, 3))  
        # 150-240h by 6h: [150, 156, 162, ..., 240] 
        max_hours = min(forecast_days * 24, 240)
        if max_hours > 144:
            extended_steps = list(range(150, max_hours + 1, 6))
            steps.extend(extended_steps)
    
    steps = [s for s in steps if s <= forecast_days * 24]
    
    
    try:
        result = client.retrieve(
            type="fc",           # forecast
            step=steps,          # forecast hours  
            param=["10u", "10v"], # 10m u and v wind components
            target=output_file,
        )
        
        print(f"✓ Download completed successfully")
        print(f"✓ Forecast base time: {result.datetime}")
        print(f"✓ File saved as: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    wind_file = download_ecmwf_wind_forecast(
        forecast_days=15,
        output_file="wind.grib2"
    )
