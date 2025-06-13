#!/usr/bin/env python3

import copernicusmarine
from datetime import datetime, timedelta

copernicusmarine.login(username="junbo.wang@uib.no", password="GPW5w*roLs@0M^")

min_lon = -55.0  # West boundary
max_lon = -15.0  # East boundary
min_lat = 35.0   # South boundary  
max_lat = 65.0   # North boundary

start_date = "2025-05-26T00:00:00"
end_date = "2025-06-05T00:00:00"  # ~9 days forecast (within 10-day limit)


copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",  # Daily SST from global physics
  variables=["thetao"],  # Sea water potential temperature
  minimum_longitude=min_lon,
  maximum_longitude=max_lon,
  minimum_latitude=min_lat,
  maximum_latitude=max_lat,
  start_datetime=start_date,
  end_datetime=end_date,
  output_filename="sst_forecast.nc"
)
