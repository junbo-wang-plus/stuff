import copernicusmarine
from datetime import datetime, timedelta

copernicusmarine.login(username="junbo.wang@uib.no", password="GPW5w*roLs@0M^")

start_date = "2025-04-21T00:00:00"
end_date = "2025-06-12T00:00:00"

min_lon = -20.0
max_lon = 20.0
min_lat = 32.0
max_lat = 72.0

copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
    variables=["eastward_wind", "northward_wind"],  
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_date,
    end_datetime=end_date,
    output_filename="wind.nc"
)

