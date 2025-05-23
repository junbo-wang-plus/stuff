import xarray as xr
from datetime import datetime, timedelta
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_grib2
from opendrift.readers.reader_netCDF_CF_generic import Reader

lon = -37; lat = 50  

o = OceanDrift(loglevel=20)
o2 = OceanDrift(loglevel=20)

reader_currents = Reader('currents_tides.nc')
reader_waves = Reader('waves.nc')
#reader_wind  = reader_grib2.Reader('wind.grib2')
#reader_wind = Reader('wind2.nc')


o.add_reader(reader_currents)
o.add_reader(reader_waves)
#o.add_reader(reader_wind)

o2.add_reader(reader_currents)

o.seed_elements(lon=lon, lat=lat, number=1000, radius=2000, time=datetime.utcnow())
o2.seed_elements(lon=lon, lat=lat, number=1000, radius=2000, time=datetime.utcnow())
#o.seed_elements(lon=lon, lat=lat, number=1000, radius=2000, time=datetime(2025,5,18,20,0,0))
o.run(duration=timedelta(days=6))
o2.run(duration=timedelta(days=6))

#o.plot(background=['x_sea_water_velocity', 'y_sea_water_velocity'])
o.plot(compare=o2,background=['x_sea_water_velocity', 'y_sea_water_velocity'],legend=['current + wave','current only'],filename='drift_comp.png')
o.animation(filename='drift.mp4',backgroud=['x_sea_water_velocity', 'y_sea_water_velocity'])
