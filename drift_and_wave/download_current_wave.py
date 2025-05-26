import copernicusmarine
from datetime import datetime, timedelta

copernicusmarine.login(username="junbo.wang@uib.no", password="GPW5w*roLs@0M^")

start_date = "2025-05-26T00:00:00"
end_date = "2025-06-05T00:00:00"

min_lon = -55.0
max_lon = -15.0
min_lat = 35.0
max_lat = 65.0

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    variables=["uo", "vo", "utide", "vtide"],  
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_date,
    end_datetime=end_date,
    output_filename="currents_tides.nc"
)

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",  
    variables=["VHM0",   
               "VMDR",   
               "VTPK",   
               "VSDX",
	       "VSDY"],  
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_date,
    end_datetime=end_date,
    output_filename="waves.nc"
)
