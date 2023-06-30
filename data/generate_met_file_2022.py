#!/usr/bin/env python

"""
Generate a CSV and netcdf met file for 2022

TODO:
=====
- Rainfall needs to be weather generated, we're divide up equally across hours
  for now.
- Data all need to be gap filled.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (29.06.2022)"
__email__ = "mdekauwe@gmail.com"

import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

def create_netcdf(lat, lon, df, out_fname):
    """
    Create a netcdf file to run LSM with. This won't work in reality until we
    gap fill...
    """
    #ndim = 1
    #n_timesteps = len(df)
    #times = []
    #secs = 0.0
    #for i in range(n_timesteps):
    #    times.append(secs)
    #    print(times)
    #    secs += 1800.

    # Bit of a faff here, but we have gaps in the timeseries so we need to
    # account for them in the date index. We really need to gapfill...
    start_date = df.index[0]
    end_date = df.index[-1]

    times = []
    for i in range(len(df)):
        secs = (df.index[i] - start_date).total_seconds()
        times.append(secs)

    ndim = 1
    n_timesteps = len(df)

    # create file and write global attributes
    f = nc.Dataset(out_fname, 'w', format='NETCDF4')
    f.description = 'Alice Holt met data, created by Martin De Kauwe'
    f.history = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date = "%s" % (datetime.now())
    f.contact = "mdekauwe@gmail.com"

    # set dimensions
    f.createDimension('time', None)
    f.createDimension('z', ndim)
    f.createDimension('y', ndim)
    f.createDimension('x', ndim)
    #f.Conventions = "CF-1.0"

    # create variables
    time = f.createVariable('time', 'f8', ('time',))
    time.units = "seconds since %s 00:00:00" % (df.index[0])
    time.long_name = "time"
    time.calendar = "standard"

    z = f.createVariable('z', 'f8', ('z',))
    z.long_name = "z"
    z.long_name = "z dimension"

    y = f.createVariable('y', 'f8', ('y',))
    y.long_name = "y"
    y.long_name = "y dimension"

    x = f.createVariable('x', 'f8', ('x',))
    x.long_name = "x"
    x.long_name = "x dimension"

    latitude = f.createVariable('latitude', 'f8', ('y', 'x',))
    latitude.units = "degrees_north"
    latitude.missing_value = -9999.
    latitude.long_name = "Latitude"

    longitude = f.createVariable('longitude', 'f8', ('y', 'x',))
    longitude.units = "degrees_east"
    longitude.missing_value = -9999.
    longitude.long_name = "Longitude"

    SWdown = f.createVariable('SWdown', 'f8', ('time', 'y', 'x',))
    SWdown.units = "W/m^2"
    SWdown.missing_value = -9999.
    SWdown.long_name = "Surface incident shortwave radiation"
    SWdown.CF_name = "surface_downwelling_shortwave_flux_in_air"

    Tair = f.createVariable('Tair', 'f8', ('time', 'z', 'y', 'x',))
    Tair.units = "K"
    Tair.missing_value = -9999.
    Tair.long_name = "Near surface air temperature"
    Tair.CF_name = "surface_temperature"

    #Rainf = f.createVariable('Rainf', 'f8', ('time', 'y', 'x',))
    #Rainf.units = "mm/s"
    #Rainf.missing_value = -9999.
    #Rainf.long_name = "Rainfall rate"
    #Rainf.CF_name = "precipitation_flux"

    Precip = f.createVariable('Precip', 'f8', ('time', 'y', 'x',))
    Precip.units = "mm/s"
    Precip.missing_value = -9999.
    Precip.long_name = "Rainfall rate"
    Precip.CF_name = "precipitation_flux"

    Qair = f.createVariable('Qair', 'f8', ('time', 'z', 'y', 'x',))
    Qair.units = "kg/kg"
    Qair.missing_value = -9999.
    Qair.long_name = "Near surface specific humidity"
    Qair.CF_name = "surface_specific_humidity"

    Wind = f.createVariable('Wind', 'f8', ('time', 'z', 'y', 'x',))
    Wind.units = "m/s"
    Wind.missing_value = -9999.
    Wind.long_name = "Scalar windspeed" ;
    Wind.CF_name = "wind_speed"

    PSurf = f.createVariable('PSurf', 'f8', ('time', 'y', 'x',))
    PSurf.units = "Pa"
    PSurf.missing_value = -9999.
    PSurf.long_name = "Surface air pressure"
    PSurf.CF_name = "surface_air_pressure"

    LWdown = f.createVariable('LWdown', 'f8', ('time', 'y', 'x',))
    LWdown.units = "W/m^2"
    LWdown.missing_value = -9999.
    LWdown.long_name = "Surface incident longwave radiation"
    LWdown.CF_name = "surface_downwelling_longwave_flux_in_air"

    CO2air = f.createVariable('CO2air', 'f8', ('time', 'z', 'y', 'x',))
    CO2air.units = "ppm"
    CO2air.missing_value = -9999.
    CO2air.long_name = ""
    CO2air.CF_name = ""

    za_tq = f.createVariable('za_tq', 'f8', ('y', 'x',))
    za_tq.units = "m"
    za_tq.missing_value = -9999.
    za_tq.long_name = "level of lowest atmospheric model layer"

    za_uv = f.createVariable('za_uv', 'f8', ('y', 'x',))
    za_uv.units = "m"
    za_uv.missing_value = -9999.
    za_uv.long_name = "level of lowest atmospheric model layer"

    # write data to file
    x[:] = ndim
    y[:] = ndim
    z[:] = ndim
    time[:] = times
    latitude[:] = lat
    longitude[:] = lon

    SWdown[:,0,0] = df.Swdown.values.reshape(n_timesteps, ndim, ndim)
    #Rainf[:,0,0] = df.Rainf.values.reshape(n_timesteps, ndim, ndim)
    Precip[:,0,0] = df.Rainf.values.reshape(n_timesteps, ndim, ndim)
    Qair[:,0,0,0] = df.Qair.values.reshape(n_timesteps, ndim, ndim, ndim)
    Tair[:,0,0,0] = df.Tair.values.reshape(n_timesteps, ndim, ndim, ndim)
    Wind[:,0,0,0] = df.Wind.values.reshape(n_timesteps, ndim, ndim, ndim)
    PSurf[:,0,0] = df.Psurf.values.reshape(n_timesteps, ndim, ndim)
    LWdown[:,0,0] = df.Lwdown.values.reshape(n_timesteps, ndim, ndim)
    CO2air[:,0,0,0] = df.CO2air.values.reshape(n_timesteps, ndim, ndim, ndim)


    # Height from Wilkinson, M., Eaton, E. L., Broadmeadow, M. S. J., and
    # Morison, J. I. L.: Inter-annual variation of carbon uptake by a
    # plantation oak woodland in south-eastern England, Biogeosciences, 9,
    # 5373–5389, https://doi.org/10.5194/bg-9-5373-2012, 2012.
    za_tq[:] = 28.  # temp - don't know measurement height
    za_uv[:] = 28.  # wind - don't know measurement height

    f.close()

def calc_esat(tair):
    """
    Calculates saturation vapour pressure

    Params:
    -------
    tair : float
        air temperature [deg C]

    Reference:
    ----------
    * Jones (1992) Plants and microclimate: A quantitative approach to
    environmental plant physiology, p110
    """

    esat = 613.75 * np.exp(17.502 * tair / (240.97 + tair))

    return esat

def estimate_lwdown(tair, rh):
    """
    Synthesises downward longwave radiation based on Tair RH

    Params:
    -------
    tair : float
        air temperature [K]
    rh : float
        relative humidity [0-1]

    Reference:
    ----------
    * Abramowitz et al. (2012), Geophysical Research Letters, 39, L04808

    """
    zeroC = 273.15

    sat_vapress = 611.2 * np.exp(17.67 * ((tair - zeroC) / (tair - 29.65)))
    vapress = np.maximum(0.05, rh) * sat_vapress
    lw_down = 2.648 * tair + 0.0346 * vapress - 474.0

    return lw_down

def qair_to_vpd(qair, tair, press):
    """
    Qair : float
        specific humidity [kg kg-1]
    tair : float
        air temperature [deg C]
    press : float
        air pressure [Pa]
    """

    PA_TO_KPA = 0.001
    HPA_TO_PA = 100.0

    # saturation vapor pressure (Pa)
    es = HPA_TO_PA * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # vapor pressure
    ea = (qair * press) / (0.622 + (1.0 - 0.622) * qair)

    vpd = (es - ea) * PA_TO_KPA

    vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd

def vpd_to_qair(vpd, tair, press):
    """
    Converts VPD to specific humidity, Qair (kg/kg)

    Params:
    --------
    VPD : float
        vapour pressure deficit [Pa]
    tair : float
        air temperature [deg C]
    press : float
        air pressure [Pa]
    """

    PA_TO_KPA = 0.001
    KPA_TO_PA = 1000.0
    HPA_TO_PA = 100.0

    tc = tair - 273.15
    # saturation vapor pressure (Pa)
    es = HPA_TO_PA * 6.112 * np.exp((17.67 * tc) / (243.5 + tc))

    # vapor pressure
    ea = es - (vpd * KPA_TO_PA)

    qair = 0.622 * ea / (press - (1 - 0.622) * ea)

    return qair

def convert_rh_to_qair(rh, tair, press):
    """
    Converts relative humidity to specific humidity (kg/kg)

    Params:
    -------
    tair : float
        air temperature [K]
    press : float
        air pressure [Pa]
    rh : float
        relative humidity [%]
    """

    tc = tair - 273.15

    # Sat vapour pressure in Pa
    esat = calc_esat(tc)

    # Specific humidity at saturation:
    ws = 0.622 * esat / (press - esat)

    # specific humidity
    qair = (rh / 100.0) * ws

    return qair

def RH_to_VPD(RH, tair):
    """
    Converts relative humidity to vapour pressure deficit (kPa)

    Params:
    --------
    RH : float
        relative humidity (%)
    tair : float
        air temperature [deg C]

    """

    PA_TO_KPA = 0.001

    # Sat vapour pressure in Pa
    esat = calc_esat(tair)

    ea = (RH / 100) * esat
    vpd = (esat - ea) * PA_TO_KPA
    vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd # kPa


def calc_esat(tair):
    """
    Calculates saturation vapour pressure

    Params:
    -------
    tair : float
        air temperature [deg C]

    Reference:
    ----------
    * Jones (1992) Plants and microclimate: A quantitative approach to
    environmental plant physiology, p110
    """

    esat = 613.75 * np.exp(17.502 * tair / (240.97 + tair))

    return esat

def Rg_to_PPFD(Rg):
    """
    Convert incoming global radiation from a pyranometer to PAR

    Params:
    -------
    Rg : float
        ncoming global radiation  [W m-2]

    """
    J_to_mol = 4.6
    frac_PAR = 0.5

    par = Rg * frac_PAR * J_to_mol
    par = np.where(par < 0.0, 0.0, par)

    return par

def simple_Rnet(tair, Swdown):
    """
    Very, very basic quick Rnet calc...see if they measured this.

    Params:
    -------
    tair : float
        air temperature [deg C]
    Swdown : float
        SW radidation [W m-2]

    """
    # Net loss of long-wave radn, Monteith & Unsworth '90, pg 52, eqn 4.17
    net_lw = 107.0 - 0.3 * tair            # W m-2

    albedo = 0.15

    # Net radiation recieved by a surf, Monteith & Unsw '90, pg 54 eqn 4.21
    #    - note the minus net_lw is correct as eqn 4.17 is reversed in
    #      eqn 4.21, i.e Lu-Ld vs. Ld-Lu
    #    - NB: this formula only really holds for cloudless skies!
    #    - Bounding to zero, as we can't have negative soil evaporation, but you
    #      can have negative net radiation.
    #    - units: W m-2
    #Rnet = MAX(0.0, (1.0 - albedo) * sw_rad - net_lw);
    Rnet = (1.0 - albedo) * Swdown - net_lw

    return Rnet

def gap_fill(df, col, interpolate=True):
    non_nans = df[col][~df[col].apply(np.isnan)]
    start, end = non_nans.index[0], non_nans.index[-1]

    if interpolate:
        df[col] = df[col].interpolate()
    else:
        df[col].loc[start:end] = df[col].loc[start:end].fillna(method='ffill')

    return df

def add_in_global_co2(co2_fname, df):

    df_co2 = pd.read_csv(co2_fname, header=59, index_col=0, parse_dates=[0])
    df_co2 = df_co2.drop('unc', axis=1)

    df['CO2air'] = -999.9 # fix later
    unique_yrs = np.unique(df.index.year)
    for yr in unique_yrs:

        years_co2 = df_co2[df_co2.index.year == yr].values[0][0]
        #print(yr, years_co2)

        idx = df.index[df.index.year==yr].tolist()
        #df['CO2air'][idx] = years_co2

        df.loc[idx, 'CO2air'] = years_co2

    return df


if __name__ == "__main__":

    #hpa_2_kpa = 0.1
    kpa_2_pa = 1000.
    deg_2_kelvin = 273.15

    # Solar radiaiton 1 W m-2 ~ 2.3 umol m-2 s-1 PAR
    # Landsberg and Sands, Cp2, pg 20. (1.0 / 2.3) #
    SW_2_PAR = 2.3
    PAR_2_SW = 1.0 / SW_2_PAR

    lat = 51.1536
    lon = -0.8582

    #fname = "../raw_data/2022/AliceHolt_2022_flux&met_ver2.csv"
    #df = pd.read_csv(fname, index_col='DateTime', parse_dates=True)
    #for col in df.columns:
    #    print(col)
    #sys.exit()

    ###
    # Read the Alice flux data, fix up the names
    ###
    fname = "../raw_data/2022/AliceHolt_2022_flux&met_ver2.csv"
    df = pd.read_csv(fname, index_col='DateTime', parse_dates=True,
                     usecols = ['DateTime','air_temperature',
                                'air_pressure', 'PAR_Den_Avg',
                                'Rn_Avg', 'u*', 'Rain_mm_Tot',
                                'VPD', 'LE', 'RH','WS_ms_Avg'])

    df = df.rename(columns={'Rain_mm_Tot': 'Rainf'})
    df = df.rename(columns={'air_pressure': 'Psurf'})
    df = df.rename(columns={'air_temperature': 'Tair'})
    df = df.rename(columns={'PAR_Den_Avg': 'PAR'})
    df = df.rename(columns={'Rn_Avg': 'Rnet'})
    df = df.rename(columns={'u*': 'Ustar'})
    df = df.rename(columns={'Rn_Avg': 'Rnet'})
    df = df.rename(columns={'WS_ms_Avg': 'Wind'})


    ###
    # Add in the global CO2
    ###
    co2_fname = "../raw_data/global_co2/co2_annmean_mlo.csv"
    df = add_in_global_co2(co2_fname, df)

    ###
    # Fill the gaps
    ###

    # Might use this for filling, work out the average hour of day
    #df_hod = df.groupby([df.index.year, df.index.hour]).agg(np.nanmean)

    # First gap fill the rainfall data with 0
    df['Rainf'] = df['Rainf'].fillna(0)

    # Account for the very, very low wind speeds as these are values but bunk
    # So, set it so we can then gap fill
    df.Wind = np.where(df.Wind < 0.1, np.nan, df.Wind)


    # Fill by the hour of day average
    df = df.groupby(df.index.hour).fillna(method='ffill')
    df = df.groupby(df.index.hour).fillna(method='bfill')


    # Gap fill the air pressure, VPD
    #df = gap_fill(df, 'Psurf', interpolate=True)
    #df = gap_fill(df, 'VPD', interpolate=True)
    #df = gap_fill(df, 'Tair', interpolate=True)

    #window = 5
    #df[ df.isnull() ] = np.nanmean( [ df.shift(x).values for x in
    #                                 range(-48*window,48*(window+1),48) ], axis=0 )

    #plt.plot(df.VPD, "r-")
    #import datetime
    #plt.xlim([datetime.date(2022, 5, 1), datetime.date(2022, 7, 29)])
    #plt.show()
    #sys.exit()


    ###
    # fix the units
    ###
    #df.loc[:, 'Rainf'] /= (48 * 1800) ## mm/sec
    df.loc[:, 'Rainf'] /= 1800 ## mm/sec

    df['Qair'] = convert_rh_to_qair(df['RH'], df['Tair'], df['Psurf'])

    #df['VPD'] = qair_to_vpd(df['Qair'], df['Tair'], df['Psurf'])
    df['VPD'] = np.where(df['VPD'] < 0.05, 0.05, df['VPD'])

    # Add LW
    df['Lwdown'] = estimate_lwdown(df.Tair.values, df.RH.values/100.)

    #df['PAR'] = Rg_to_PPFD(df['Rg'])
    df['Swdown'] = df['PAR'] * PAR_2_SW
    #df['Rnet'] = simple_Rnet(df['Tair'], df['Swdown'])

    df.to_csv("alice_holt_met_data_2022.csv")

    out_fname = "UK-Ham_2002-2003_Met.nc"
    create_netcdf(lat, lon, df, out_fname)
