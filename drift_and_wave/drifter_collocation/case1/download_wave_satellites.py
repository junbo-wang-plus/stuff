import copernicusmarine
from datetime import datetime, timedelta

copernicusmarine.login(username="junbo.wang@uib.no", password="GPW5w*roLs@0M^")

start_date = "2025-04-22T00:00:00"
end_date = "2025-06-11T00:00:00"

min_lon = -22.0
max_lon = 20.0
min_lat = 30.0
max_lat = 72.0

product_names = [
	"cmems_obs-wave_glo_phy-swh_nrt_cfo-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_c2-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_h2b-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_h2c-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_j3-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_al-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_s3a-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_s3b-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_s6a-l3_PT1S",
	"cmems_obs-wave_glo_phy-swh_nrt_swon-l3_PT1S"]


output_names = ["cfo","c2","h2b","h2c","j3",
	"al","s3a","s3b","s6a","swon"]


for i, (product_id, output_file) in enumerate(zip(product_names, output_names), 1):
    
	copernicusmarine.subset(
	    dataset_id=product_id,
	    variables=["VAVH"],  # Common variable name for significant wave height
	    minimum_longitude=min_lon,
	    maximum_longitude=max_lon,
	    minimum_latitude=min_lat,
	    maximum_latitude=max_lat,
	    start_datetime=start_date,
	    end_datetime=end_date,
	    output_filename=output_file
	)
        
