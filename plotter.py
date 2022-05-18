import xarray as xr
from parsivel_utils import plot_dsd_timeseries

ds1 = xr.open_dataset('parsivel_data_2022-05-09.nc')
ds2 = xr.open_dataset('parsivel_data_2022-05-10.nc')
ds = xr.concat([ds1, ds2],dim='time')
plot_dsd_timeseries(ds)