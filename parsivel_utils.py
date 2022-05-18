"""
    Parse data from parsivel text file
    
    Raul Valenzuela
    UOH
    2022

"""

from csv import reader
import numpy as np
import xarray as xr
import pandas as pd

import proplot as plot
import matplotlib.pyplot as plt

file_velocity_bins = 'parsivel_bins_velocity.csv'
file_diameter_bins = 'parsivel_bins_diameter.csv'

def parsivel2nc(lines, idx_group, timestamp_group, nc_dir, nc_file_fmt):

    ''' read parsivel bins '''
    velo_bins = pd.read_csv(file_velocity_bins, delimiter=';')
    diam_bins = pd.read_csv(file_diameter_bins, delimiter=';')       

    
    ''' find index of first "<" appearance in each string to separate data and spectrum part '''
    lines_group = [lines[i_g] for i_g in idx_group]
    idx_s = list(map(lambda x:x.find('<'), lines_group))

    ''' get the spectrum part '''
    all_spectrum = list(map(lambda s,i:s[i:], lines_group, idx_s))

    ''' get all rows with valid spectrum values '''
    non_zero_idx = list(map(lambda x: x[0] if 'ZERO' not in x[1] else None, enumerate(all_spectrum)))

    ''' create cube with spectrum data '''
    first = True
    for idx in non_zero_idx:
        if idx is None:
            array = np.empty((32,32,1))
            array[:] = np.nan
        else:
            sp = all_spectrum[idx]
            values = sp.split(';')[:-1]
            values_parsed = ['0' if (x == '' or x == '<SPECTRUM>') else x for x in values]    
            array = np.array(values_parsed).reshape((32,32)).astype('float64')
            array = array[:,:,np.newaxis]

        if first:
            cube = array.copy()
            first = False
        else:
            cube = np.concatenate([cube, array], axis=2)    

    ''' create xarray with cube and timestamp '''
    da = xr.DataArray(data=cube, 
                       dims=['velocity','diameter','time'],
                       coords={'time':list(timestamp_group),
                               'velocity':velo_bins['average'],
                               'diameter':diam_bins['average']
                              }
                      )            

    ''' get the meteorological part '''
    data0 = list(map(lambda s,i:s[20:i-1],lines_group,idx_s))

    ''' replace comma by dot in decimal place '''
    data = list(map(lambda s: s.replace(',','.'), data0))

    ''' split data into columns '''
    met_vars = list(map(lambda x: tuple(x.split(';')), data))
    met_vars = list(zip(*met_vars))

    ds = xr.Dataset(
                data_vars=dict(
                    pcp_rate=(["time"], np.array(met_vars[0]).astype('float'),
                             {"units": "mm h-1"}),
                    n_particles=(["time"], np.array(met_vars[1]).astype('int'),
                                {"units": "number"}),
                    t_in_sensor=(["time"], np.array(met_vars[2]).astype('float'),
                                {"units": "ºC"}),
                    snow_rate=(["time"], np.array(met_vars[3]).astype('float'),
                              {"units": "mm h-1"}),
                    pcp_since_start=(["time"], np.array(met_vars[4]).astype('float'),
                                    {"units": "mm"}),
#                     reflectivity=(["time"], np.array(met_vars[5]).astype('float'),
#                                  {"units": "dBZ"}),
                ),
                coords=dict(
                    time=(["time"], list(timestamp_group)),
                ),
                attrs=dict(
                    description="Parsivel optical disdrometer data from ARO-UdeC"),
        )

    ds['spectrum'] = da
    nc_out = timestamp_group[0].strftime(nc_dir+nc_file_fmt)
    print('Saving {}'.format(nc_out))
    ds.to_netcdf(nc_out)

def plot_spectrum():
    pass

def plot_dsd_timeseries(ds):
    
    dsd = np.log10(ds['spectrum'].sum(dim='velocity'))
#     dsd_masked = dsd.where(dsd != np.inf) 
    
    precip = ds['pcp_rate']
    
    fig = plot.figure(figsize=(8,6), sharex=True, sharey=False, suptitle='Parsivel at ARO-UdeC')

    ax = fig.subplot(211)
    m = ax.pcolormesh(dsd, cmap='viridis')
    ax.set_facecolor((.95, .9, .95))
    ax.format(
        title='Drop Size Distribution in Time',
        xlabel='Time', ylabel='drop diameter [mm]',
        ylim=(0,10),
        )
    ax.colorbar(m, loc='r',label='log10(N)')

    ax = fig.subplot(212)
    ax.plot(precip)
    ax.format(
        title='Rain Intensity',
        xlabel='Time', ylabel='rain rate [mm/h]',
        xformatter='concise',
        xrotation=0,
        ylim=(0,20)
        )

    plt.savefig('parsivel_dsd_timeserie_last.png')
    