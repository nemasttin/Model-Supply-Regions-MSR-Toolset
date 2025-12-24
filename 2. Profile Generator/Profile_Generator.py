import pandas as pd
from netCDF4 import Dataset
import numpy as np
import itertools
from scipy import spatial
from scipy.interpolate import interp1d
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pvlib
import re
import math
import geopandas as gpd
import time
import traceback

def ReadERA5Data(ERA5Data_Inst_folder, ERA5Data_Acc_folder, ERA5Data_startyear, ERA5Data_endyear):

    years = list(range(ERA5Data_startyear, ERA5Data_endyear + 1))
    np_ERA5Data_Inst_dict = {}
    np_ERA5Data_Acc_dict = {}

    for year in years:
        ERA5Data_Inst_path = os.path.join(ERA5Data_Inst_folder, f'inst_{year}_06.nc')
        ERA5Data_Acc_path = os.path.join(ERA5Data_Acc_folder, f'acc_{year}_06.nc')

        if not os.path.isfile(ERA5Data_Inst_path):
            raise FileNotFoundError(f"Missing: {ERA5Data_Inst_path}")
        if not os.path.isfile(ERA5Data_Acc_path):
            raise FileNotFoundError(f"Missing: {ERA5Data_Acc_path}")
        
        np_ERA5Data_Inst_dict[year] = Dataset(ERA5Data_Inst_path)
        np_ERA5Data_Acc_dict[year] = Dataset(ERA5Data_Acc_path)
    
    return np_ERA5Data_Inst_dict, np_ERA5Data_Acc_dict

# ComputeHourlyCF_SolarPV
def ComputeHourlyCF_SolarPV (GSA_GHI_MSR_Mean, temp_2m, GHI): 
    
    # Compute hourly solar PV capacity factors for a single location (MSR centroid).

    # Inputs
        # GHI: np.ndarray
        #   ERA5 hourly surface solar radiation downwards (ssrd) [J/m2].
        # temp_2m: np.ndarray
        #   ERA5 hourly 2 m air temperature [Kelvin].
        # GSA_GHI_MSR_Mean: float
        #   Global Solar Atlas (GSA) long-term mean GHI at 1 km resolution averaged across the MSR [kWh/m2/day].
    # Outputs
        # CF_solar: np.ndarray
        #   Hourly solar PV capacity factor [-].
        # GHI_corrected_wh: np.ndarray
        #   Bias-corrected hourly GHI [Wh/m2].
        # BiasCorrection_GHI_Adder_Wh: float
        #   Bias correction added to GHI [Wh/m2].
        # ERA5_GHI_OriginalAnnualYield: float
        #   ERA5 annual GHI yield before correction [kWh/m2].


    GHI=np.ma.filled(GHI)
    temp_2m=np.ma.filled(temp_2m)

    # Remove negative irradiance values
    GHI[GHI < 0] = 0

    # Convert ERA GHI given in J/m2 to kWh/m2
    GHI=GHI/3600000
    ERA5_GHI_OriginalAnnualYield=np.sum(GHI)

    # Bias correction
    GHIAnnualBias_KWh=GSA_GHI_MSR_Mean*(TimeSteps/24)-ERA5_GHI_OriginalAnnualYield
    GHI_ifallCorrected = GHI + GHIAnnualBias_KWh / TimeSteps
    BiasCorrection_GHI_Adder_KWh=GHIAnnualBias_KWh/len(GHI[(GHI!=0)&(GHI_ifallCorrected<1)&(GHI_ifallCorrected>0)])
    GHI[(GHI!=0)&(GHI_ifallCorrected<1)&(GHI_ifallCorrected>0)]=GHI[(GHI!=0)&(GHI_ifallCorrected<1)&(GHI_ifallCorrected>0)]+BiasCorrection_GHI_Adder_KWh
    GHI[GHI<0]=0

    # Convert back to Wh/m2 for PV performance model
    GHI_corrected_wh = GHI*1000
    BiasCorrection_GHI_Adder_Wh=BiasCorrection_GHI_Adder_KWh * 1000

    # PV performance model

    Gstc = 1000 # [W/m2]
    Tmod_stc = 25 +273 # [K]
    k = [-0.017162, -0.040289, -0.004681, 0.000148, 0.000169, 0.000005]
    cT = 0.035 # [1/K]
    GHI_Wh=GHI_corrected_wh # Yunshu code uses W/m2
    G_norm = GHI_Wh/Gstc
    T_norm = cT*GHI_Wh + temp_2m-Tmod_stc

    log_G_norm = np.zeros_like(G_norm, dtype=float)
    mask = GHI_Wh != 0
    log_G_norm[mask] = np.log(G_norm[mask])

    n_rel = 1 + k[0]*log_G_norm + k[1]*(log_G_norm)**2 + T_norm*(k[2] + k[3]*log_G_norm + k[4]*(log_G_norm)**2) + k[5]*T_norm**2

    CF0 = n_rel * G_norm
    CF_solar = np.nan_to_num(CF0)
    CF_solar[CF_solar<0]=0

    return CF_solar, GHI_corrected_wh, BiasCorrection_GHI_Adder_Wh, ERA5_GHI_OriginalAnnualYield

# ComputeHourlyCF_Wind
def ComputeHourlyCF_Wind(BiasCorrEffectiveWindSpeeds, u100, u10, v100, v10, WindTurbineHeight_meters, temp_2m, elevation, pd_WindSpeed_to_Power):# returns wind hourly CF
    
    # Compute hourly wind capacity factors for a single location (MSR centroid)

    # Inputs
        # BiasCorrEffectiveWindSpeeds: np.ndarray
        #   ERA5 bias-corrected effective wind speeds at 100 m [m/s].
        # u100, v100, u10, v10: np.ndarray
        #   ERA5 wind speed components needed to get wind speeds and resulting CFs at any desired height [m/s].
        # WindTurbineHeight_meters: float
        #   Desired hub height to get ERA5 resource profile (biascorrected at 100m) and CFs.
        # temp_2m: np.ndarray
        #   ERA5 2 m air temperature [K].
        # elevation: float
        #   MSR elevation above sea level [m] from ERA5, needed to get location's air density that would adjust wind speeds as proxy to impact of air density on wind CF.
        # pd_WindSpeed_to_Power: pd.DataFrame
        #   Wind turbine power curves for three IEC classes.
    # Outputs
        # CF_wind: np.ndarray
        #   Height and air density adjusted wind capacity factors [-].
        # map_windspeed_hubh: np.ndarray
        #   Hub-height wind speeds [m/s] before air density adjustment.
        # ERA5_AnnualMean100m: float
        #   Annual mean wind speed at 100 m from ERA5 [m/s] (debugging purpose).

    h_0 = 100.0 # Reference extrapolation height where bias correction is applied
    h_hub = WindTurbineHeight_meters

    # Calculate pressure using IRENA-LBNL report, units mentioned there, temperature from ERA5 is in kelvins with no zeros
    rho_0 = 1.225  # kg/m^3
    R = 287.058  # J/(kg.K)

    pressure_ts = 101325 / np.exp(elevation * 9.80665 / (R * temp_2m))
    LocationDensity_ts = pressure_ts / (R * temp_2m)

    # Calculate adjust wind speeds as per ACEC methodology
    map_windspeed_10m = (u10 ** 2 + v10 ** 2) ** (1 / 2)
    map_windspeed_100m = (u100 ** 2 + v100 ** 2) ** (1 / 2)

    ERA5_AnnualMean100m = np.mean(map_windspeed_100m)
    map_windspeed_100mCorrected = BiasCorrEffectiveWindSpeeds
    map_windspeed_100m_UnCorrected = map_windspeed_100m #only for diagnosis

    alpha = np.log(np.nan_to_num(map_windspeed_100m/map_windspeed_10m, posinf=0, neginf=0)) / np.log(100 / 10)#for alpha dont use corrected speeds
    if h_hub==h_0:
        map_windspeed_hubh=map_windspeed_100mCorrected
        map_windspeed_hubh_Uncorrected = map_windspeed_100m_UnCorrected #only for diagnosis
    else:
        map_windspeed_hubh = map_windspeed_100mCorrected * ((h_hub / h_0)** alpha)
        map_windspeed_hubh_Uncorrected = map_windspeed_100m_UnCorrected * ((h_hub / h_0) ** alpha) #only for diagnosis

    # Uplifts or downlifts the annual mean a bit from GWA
    map_windspeed_hubh_adjusted = map_windspeed_hubh * (rho_0 / LocationDensity_ts) ** (1 / 3)

    # Determine IEC class based on annual mean hub height wind speed
    win_avg_speed = map_windspeed_hubh.mean()
    win_class_bounds = [7.5, 8.5]
    if win_avg_speed <= win_class_bounds[0]:
        IEC_Class = 'IEC Class 3'
    elif win_class_bounds[0] <= win_avg_speed < win_class_bounds[1]:
        IEC_Class = 'IEC Class 2'
    elif win_avg_speed > win_class_bounds[1]:
        IEC_Class = 'IEC Class 1'
    else:
        raise Exception('unidentified speed detected')


    f = interp1d(pd_WindSpeed_to_Power.iloc[:, 0],pd_WindSpeed_to_Power[IEC_Class], fill_value = "extrapolate")
    # plt.figure(1)
    # plt.plot(np.arange(0, 25, 0.1),f(np.arange(0, 25, 0.1)))
    CF_wind = f(np.ma.filled(map_windspeed_hubh_adjusted[:]))

    return CF_wind, map_windspeed_hubh, map_windspeed_hubh_Uncorrected, ERA5_AnnualMean100m

# develop_allMSR_8760BiasCorrEffectiveWindSpeeds_fityear
def develop_allMSR_8760BiasCorrEffectiveWindSpeeds_fityear(pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear, pd_Wind_GWA_MSR_Mean, TimeSteps):
    
    # Bias-correct hourly ERA5 effective wind speed profiles for all MSRs.

    # Inputs
        # pd_allMSR_Hourly100mEffectiveWindSpeeds: pd.DataFrame (nMSR, TimeSteps)
        #   Rows correspond to MSRs, columns correspond to hours (1 to TimeSteps). ERA5 effective wind speeds at 100 m before bias correction [m/s].
        # pd_Wind_GWA_MSR_Mean: pd.Series
        #   Global Wind Atlas (GWA) long-term mean wind speed at 1 km resolution averaged across the MSR [m/s].
        # TimeSteps: int
        #   Number of time steps in the time series.
    # Outputs
        # a: np.ndarray (TimeSteps,)
        #   Slope a of the linear fit for each rank.
        # b: np.ndarray (TimeSteps,)
        #   Intercept b of the linear fit for each rank.
        # S0_fit: pd.DataFrame (nMSR, TimeSteps)
        #   Sorted ERA5 wind speed per MSR before bias correction [m/s].
        # Shat0_fit: pd.DataFrame (nMSR, TimeSteps)
        #   Sorted ERA5 wind speed per MSR after bias correction [m/s].
        # pd_allMSR_TimeSeriesProfilesCorrected: pd.DataFrame (nMSR, TimeSteps)
        #   Bias-corrected ERA5 wind speed time series per MSR [m/s].

    V0 = pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.to_numpy(dtype=float)                        # (nMSR, TimeSteps)
    gwa_mean = pd_Wind_GWA_MSR_Mean.loc[pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index]

    # Create ranks (1,2...TimeSteps). Rank 1 goes to the smallest wind speed.
    ranks = [i + 1 for i in range(TimeSteps)]                                                         # (TimeSteps,)

    # Single MSR case: simple linear scaling
    if V0.shape[0] == 1:
    
        S0 = np.sort(V0, axis=1)                                                                      # (1, TimeSteps)
        scale = float(gwa_mean/V0.mean(axis=1))
        Shat0 = S0 * scale

        order = np.argsort(V0, axis=1, kind="mergesort")
        inv_order = np.empty_like(order)
        inv_order[0, order[0, :]] = np.arange(TimeSteps)
        corr = np.take_along_axis(Shat0, inv_order, axis=1)

        S0_fityear = pd.DataFrame(S0, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index, columns=ranks)
        Shat0_fityear = pd.DataFrame(Shat0, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index, columns=ranks)
        pd_allMSR_TimeSeriesProfilesCorrected_fityear = pd.DataFrame(corr, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index, columns=ranks)

        return S0_fityear, Shat0_fityear, pd_allMSR_TimeSeriesProfilesCorrected_fityear
    
    # Phase 1: Rank wise fitting and extrapolation

    # Compute the dependent variable x i.e. ERA5 annual mean wind speed of each MSR.
    x = V0.mean(axis=1)                                                                               # (nMSR,)

    # Compute the independent variable y i.e. sorted ERA5 wind speed of each MSR.
    # Sort each MSR row in ascending order to obtain order statistics.
    S0 = np.sort(V0, axis=1)                                                                          # (nMSR, TimeSteps)
    S0_fityear = pd.DataFrame(S0, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index, columns=ranks)

    # Vectorised linear fitting
    # For each rank r, determine a and b by fitting y_r = a*x + b across MSRs.
    #   x: ERA5 annual mean wind speed of each MSR.
    #   y: sorted ERA5 wind speed of each MSR for each rank.

    x_mean = x.mean()                                                    
    x_center = x - x_mean
    den = np.sum(x_center ** 2)

    y = S0                                                                                      
    y_mean = y.mean(axis=0)                                   
    y_center = y - y_mean                                      

    a = (x_center[:, None] * y_center).sum(axis=0) / den         
    b = y_mean - a * x_mean                                    

    # For each rank r, compute the bias-corrected wind speed across MSRs using the linear fit parameters a and b.
    gwa = gwa_mean.to_numpy(dtype=float)                                                               # (nMSR,)
    Shat0 = gwa[:, None] * a[None, :] + b[None, :]
    Shat0_fityear = pd.DataFrame(Shat0, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index, columns=ranks)

    # Phase 2: Mapping corrected ranks back to time series
    
    # For each MSR (row), compute the indices that would sort the original time series.
    # order[i, k] = original time index of the k-th smallest element in MSR i.
    order = np.argsort(V0, axis=1, kind="mergesort")           # (nMSR, TimeSteps)

    # inv_order[i, t] = rank position of time index t in the sorted time series of MSR i.
    inv_order = np.empty_like(order)
    inv_order[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]

    # Use inv_order to map the sorted corrected profiles back to the original time series order. corr[i,t] = Shat0[i,k] where k is the rank position of time index t in MSR i.
    corr = np.take_along_axis(Shat0, inv_order, axis=1)          # (nMSR, TimeSteps)

    # Return corrected time series profiles as DataFrame
    pd_allMSR_TimeSeriesProfilesCorrected_fityear = pd.DataFrame(corr, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.index, columns=pd_allMSR_Hourly100mEffectiveWindSpeeds_fityear.columns)

    return S0_fityear, Shat0_fityear, pd_allMSR_TimeSeriesProfilesCorrected_fityear

# develop_allMSR_8760BiasCorrEffectiveWindSpeeds_year
def develop_allMSR_8760BiasCorrEffectiveWindSpeeds_year(pd_allMSR_Hourly100mEffectiveWindSpeeds_year, S0_fityear, Shat0_fityear, TimeSteps):

    Vy = pd_allMSR_Hourly100mEffectiveWindSpeeds_year.to_numpy(dtype=float)                                     # (nMSR, TimeSteps)

    ranks = [i + 1 for i in range(TimeSteps)]                                                                   # (TimeSteps,)

    S0 = S0_fityear.loc[pd_allMSR_Hourly100mEffectiveWindSpeeds_year.index, ranks].to_numpy(dtype=float)        # (nMSR, TimeSteps)
    Shat0 = Shat0_fityear.loc[pd_allMSR_Hourly100mEffectiveWindSpeeds_year.index, ranks].to_numpy(dtype=float)  # (nMSR, TimeSteps)

    # Single MSR case: simple linear scaling
    if Vy.shape[0] == 1:

        Sy = np.sort(Vy, axis=1)                                                                      # (1, TimeSteps)
        Shaty = np.interp(Sy[0,:], S0[0,:], Shat0[0,:], left=Shat0[0,0], right=Shat0[0,-1])[None, :]

        order = np.argsort(Vy, axis=1, kind="mergesort")
        inv_order = np.empty_like(order)
        inv_order[0, order[0, :]] = np.arange(TimeSteps)
        corr = np.take_along_axis(Shaty, inv_order, axis=1)
        pd_allMSR_TimeSeriesProfilesCorrected_year = pd.DataFrame(corr, index=pd_allMSR_Hourly100mEffectiveWindSpeeds_year.index, columns=pd_allMSR_Hourly100mEffectiveWindSpeeds_year.columns)
        
        return pd_allMSR_TimeSeriesProfilesCorrected_year
    
    # Phase 1: Sorting original time series
    Sy = np.sort(Vy, axis=1)                                            

    # Phase 2: 
    idx = np.vstack([np.searchsorted(S0[i], Sy[i], side="left") for i in range(S0.shape[0])])                         

    k = idx - 1
    k = np.clip(k, 0, TimeSteps - 2)                               

    S0_left = np.take_along_axis(S0, k, axis=1)                        
    S0_right = np.take_along_axis(S0, k + 1, axis=1)                   

    denom = np.maximum(S0_right - S0_left, 1e-12)                    
    alpha = (Sy - S0_left) / denom                   

    Shat0_left = np.take_along_axis(Shat0, k, axis=1)
    Shat0_right = np.take_along_axis(Shat0, k + 1, axis=1)

    Shaty = (1 - alpha) * Shat0_left + alpha * Shat0_right

    # Clamp tails
    below_min = (idx == 0)
    above_max = (idx == TimeSteps)

    Shaty = np.where(below_min, Shat0[:, [0]], Shaty)
    Shaty = np.where(above_max, Shat0[:, [-1]], Shaty)

    # Phase 3: Mapping corrected ranks back to time series

    # For each MSR (row), compute the indices that would sort the original time series.
    # order[i, k] = original time index of the k-th smallest element in MSR i.
    order = np.argsort(Vy, axis=1, kind="mergesort")           # (nMSR, TimeSteps)

    # inv_order[i, t] = rank position of time index t in the sorted time series of MSR i.
    inv_order = np.empty_like(order)
    inv_order[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]

    # Use inv_order to map the sorted corrected profiles back to the original time series order.
    corr = np.take_along_axis(Shaty, inv_order, axis=1)          # (nMSR, TimeSteps)

    # Return corrected time series profiles as DataFrame
    pd_allMSR_TimeSeriesProfilesCorrected_year = pd.DataFrame(
        corr,
        index=pd_allMSR_Hourly100mEffectiveWindSpeeds_year.index,
        columns=pd_allMSR_Hourly100mEffectiveWindSpeeds_year.columns
    )

    return pd_allMSR_TimeSeriesProfilesCorrected_year

def CreateLocalTimeProfile(pd_UTC, pd_CountryUTC_offsets, country_withspaces):

    # Convert UTC hourly profiles to local time hourly profiles for a given country.

    # Inputs
        # pd_UTC: pd.DataFrame
        # pd_CountryUTC_offsets: pd.DataFrame
        # country_withspaces: str
    # Outputs
        # pd_LocalTime: pd.DataFrame
 
    # Create Local Time Profile
    UTCHourTags = [f'H{i}' for i in range(1, TimeSteps + 1)]
    offset = pd_CountryUTC_offsets[pd_CountryUTC_offsets.Country == country_withspaces].Hours.iloc[0]
    LocalTimeHourTags = [f'H{(i % TimeSteps) + 1}' for i in range(offset, TimeSteps + offset)]
    ColRenameDictionary = dict(zip(UTCHourTags, LocalTimeHourTags))
    print(f"{country_withspaces} UTC Offset:{offset}")
    # print(ColRenameDictionary)

    pd_LocalTime = pd_UTC.rename(columns=ColRenameDictionary)
    pd_LocalTime = pd_LocalTime.set_index(pd_LocalTime.MSR_ID[:])
    pd_LocalTime = pd_LocalTime.drop(['MSR_ID'], axis=1)

    cols = pd_LocalTime.columns.to_list()
    if offset > 0:
        NewColOrder = cols[:-TimeSteps] + cols[-offset:] + cols[-TimeSteps:-offset]
        pd_LocalTime = pd_LocalTime[NewColOrder]
    if offset < 0:
        NewColOrder = cols[:-TimeSteps] + cols[-TimeSteps - offset:] + cols[-TimeSteps:-TimeSteps - offset]
        pd_LocalTime = pd_LocalTime[NewColOrder]

    return pd_LocalTime

# Start of the main program.

# Read control input file
ControlPathsAndNames=pd.read_excel('ControlFile_ProfileGenerator.xlsx', sheet_name="PathsAndNames", index_col=0)
ControlConfigurations=pd.read_excel('ControlFile_ProfileGenerator.xlsx', sheet_name="Configurations", index_col=0)
pd_CountryUTC_offsets=pd.read_excel('ControlFile_ProfileGenerator.xlsx', sheet_name="CountryUTC_Offset_InUse")

# Load paths
ERA5Data_Inst_folder=ControlPathsAndNames.loc["ERA5DataFilePath_Inst"][0]
ERA5Data_Acc_folder=ControlPathsAndNames.loc["ERA5DataFilePath_Acc"][0]
ERA5Data_fityear=int(ControlPathsAndNames.loc["ERA5Data_fityear"][0])
ERA5Data_startyear=int(ControlPathsAndNames.loc["ERA5Data_startyear"][0])
ERA5Data_endyear=int(ControlPathsAndNames.loc["ERA5Data_endyear"][0])
pd_WindSpeed_to_Power = pd.read_csv(ControlPathsAndNames.loc["ThreeIEC_TurbinePowerCurves"][0])
AllCountries=pd.read_csv(ControlPathsAndNames.loc["FileAddress_CountryNamesList"][0],names=["Ct"])
Input_MSR_Folder=ControlPathsAndNames.loc["Input_MSR_Folder"][0]
OutputFolder_UTCProfiles=os.path.join(ControlPathsAndNames.loc["OutputFolder"][0], "Results_UTC_Profiles")
OutputFolder_LocalTime=os.path.join(ControlPathsAndNames.loc["OutputFolder"][0], "Results_LocalTimeProfiles")
ResourceRasterCarryingSubFolderName=ControlPathsAndNames.loc["ResourceRasterCarryingSubFolderName"][0]
MSR_DataCarryingSubFolderName=ControlPathsAndNames.loc["MSR_DataCarryingSubFolderName"][0]
SolarPVNameConvention=ControlPathsAndNames.loc["SolarPVNameConvention"][0]
WindNameConvention=ControlPathsAndNames.loc["WindNameConvention"][0]
TimeSteps=int(ControlPathsAndNames.loc["TimeSteps"][0])

np_ERA5Data_Inst_dict, np_ERA5Data_Acc_dict = ReadERA5Data(ERA5Data_Inst_folder, ERA5Data_Acc_folder, ERA5Data_startyear, ERA5Data_endyear)

print("Inputs loaded from control file")
print(f"Creating time series profiles for country: {AllCountries['Ct'].tolist()}")
print(f"ERA5 instantaneous variables{list(np_ERA5Data_Inst_dict[ERA5Data_fityear].variables.keys())}")
print(f"ERA5 accumulated variables{list(np_ERA5Data_Acc_dict[ERA5Data_fityear].variables.keys())}")

inst_steps = len(np_ERA5Data_Inst_dict[ERA5Data_fityear].dimensions["valid_time"])
acc_steps = len(np_ERA5Data_Acc_dict[ERA5Data_fityear].dimensions["valid_time"])
if int(TimeSteps) != inst_steps or int(TimeSteps) != acc_steps:
    raise ValueError(
        f"TimeSteps mismatch: ControlFile TimeSteps={int(TimeSteps)},"
        f"ERA5 Inst timesteps={inst_steps}, ERA5 Acc timesteps{acc_steps}")

# Load configurations
RE_TechnologyList=[] # naming as per three technology names for which MSR creator code creates MSRs
ResourceRasterNameList=[]

if ControlConfigurations.loc["Run code for SolarPV"][0]==1:
    RE_TechnologyList.append(SolarPVNameConvention)
    ResourceRasterNameList.append(ControlPathsAndNames.loc["SolarPV_ResourceRasterName"][0])
if ControlConfigurations.loc["Run code for Wind"][0]==1:
    RE_TechnologyList.append(WindNameConvention)
    ResourceRasterNameList.append(ControlPathsAndNames.loc["Wind_ResourceRasterName"][0])
    WindTurbineHeight_meters = ControlConfigurations.loc["For wind, give wind turbine height (meters)"][0]

flag_Diagnosis=ControlConfigurations.loc["Produce hourly resource profiles for diagnostics"][0]

# Code run time flags
flag_RunBiasCorrCode = 1  # Not a user option
pd_LogFile=pd.DataFrame()
DateTimeStamp = time.localtime()
DateTimeStamp = f"{DateTimeStamp.tm_year}{DateTimeStamp.tm_mon}{DateTimeStamp.tm_mday}{DateTimeStamp.tm_hour}{DateTimeStamp.tm_min}{DateTimeStamp.tm_sec}"

if not os.path.isdir(OutputFolder_UTCProfiles):
    os.makedirs(OutputFolder_UTCProfiles)
if not os.path.isdir(OutputFolder_LocalTime):
    os.makedirs(OutputFolder_LocalTime)

S0_fityear = Shat0_fityear = None
years = [ERA5Data_fityear] + [y for y in range(ERA5Data_startyear, ERA5Data_endyear + 1) if y != ERA5Data_fityear]

for year in years:

    np_ERA5Data_Inst = np_ERA5Data_Inst_dict[year]
    np_ERA5Data_Acc = np_ERA5Data_Acc_dict[year]
    print(f"Processing ERA5 year {year}")

    for CountryCounter in range(0,len(AllCountries)):   # country wise loop
        country_withspaces=AllCountries.Ct[CountryCounter]
        country = AllCountries.Ct[CountryCounter].replace(" ", "")
        MSR_CountryFolder=os.path.join(Input_MSR_Folder, country)

        OutputCountryFolder_UTC=os.path.join(OutputFolder_UTCProfiles, country)
        if not os.path.isdir(OutputCountryFolder_UTC):
            os.makedirs(OutputCountryFolder_UTC)

        OutputCountryFolder_LocalTime = os.path.join(OutputFolder_LocalTime, country)
        if not os.path.isdir(OutputCountryFolder_LocalTime):
            os.makedirs(OutputCountryFolder_LocalTime)

        for TechCounter in range (0,len(RE_TechnologyList)):
            RE=RE_TechnologyList[TechCounter]
            ResourceRasterName=ResourceRasterNameList[TechCounter]


            # Read RE MSR information
            try:
                shapefile_path = os.path.join(MSR_CountryFolder, MSR_DataCarryingSubFolderName, f"{RE}_FinalMSRs.shp")
                raster_path = os.path.join(MSR_CountryFolder, ResourceRasterCarryingSubFolderName, f"{ResourceRasterName}_projected.tif")
                gpd_MSR_Attributes = gpd.read_file(shapefile_path)
                # Get stats of resource value across each MSR as Json dictionary (dc)
                dc_ResourceStatsAcrossMSR = zonal_stats(
                                        shapefile_path,
                                        raster_path,
                                        stats="count min mean max median sum")
                gpd_MSR_Attributes['Longitude'] =  gpd_MSR_Attributes.to_crs('EPSG:4326').centroid.x
                gpd_MSR_Attributes['Latitude'] =  gpd_MSR_Attributes.to_crs('EPSG:4326').centroid.y
                gpd_MSR_Attributes = gpd_MSR_Attributes.drop(['geometry'], axis=1)


                # Find nearest neighbors of MSR coordinates in ERA5 data set
                ERA5Locations=list(itertools.product(np_ERA5Data_Inst.variables["latitude"][:],np_ERA5Data_Inst.variables["longitude"][:]))
                MSR_CentroidLocations=list(zip(gpd_MSR_Attributes.Latitude,gpd_MSR_Attributes.Longitude))
                ERA5Grid = spatial.cKDTree(ERA5Locations)
                dist, indexes = ERA5Grid.query(MSR_CentroidLocations)
                ERA5Locations_near_MSR_CentroidLocations= np.array(ERA5Locations)[indexes]
                gpd_MSR_Attributes["ERA5Latitude"]=ERA5Locations_near_MSR_CentroidLocations[:,0]
                gpd_MSR_Attributes["ERA5Longitude"]=ERA5Locations_near_MSR_CentroidLocations[:,1]
                np_lon = np_ERA5Data_Inst.variables["longitude"][:]
                np_lat = np_ERA5Data_Inst.variables["latitude"][:]


                # Declare variables to run subprograms that get capacity factor profiles from resource values for each MSR
                np_allMSR_HourlyCF=np.zeros((len(gpd_MSR_Attributes),TimeSteps))

                np_allMSR_HourlyWindSpeeds_corrected_ResultantVector = np.zeros((len(gpd_MSR_Attributes), TimeSteps)) # [m/s]
                np_allMSR_HourlyWindSpeeds_uncorrected_ResultantVector = np.zeros((len(gpd_MSR_Attributes), TimeSteps))  # [m/s]
                np_allMSR_HourlyGHI = np.zeros((len(gpd_MSR_Attributes), TimeSteps)) # [Wh]
                np_allMSR_HourlyGHI_corrected_Wh=np.zeros((len(gpd_MSR_Attributes), TimeSteps))

                # Initialize datasets for wind bias correction
                np_allMSR_ERA5_AnnualMean=np.zeros(len(gpd_MSR_Attributes))
                np_allMSR_Wind_GWA_MSR_Mean = np.zeros(len(gpd_MSR_Attributes))

                # Initialize datasets for solar bias correction
                np_allMSR_GHI_ERA5OriginalAnnualYield = np.zeros(len(gpd_MSR_Attributes))
                np_allMSR_GHI_GSA_MSR_Mean = np.zeros(len(gpd_MSR_Attributes))
                np_allMSR_BiasCorrection_GHI_Adder_Wh = np.zeros(len(gpd_MSR_Attributes))

                if flag_RunBiasCorrCode and RE==WindNameConvention:
                    # Get effective windspeeds, 8760 Bias Corrected
                    # Make all MSR timeseries profiles
                    pd_allMSR_Hourly100mEffectiveWindSpeeds=pd.DataFrame()
                    rows = []
                    for MSR_Counter in range(0, len(gpd_MSR_Attributes)): # MSR wise loop
                        print(f"appending effective ERA5 speed dataset MSR {MSR_Counter}")

                        lat = ERA5Locations_near_MSR_CentroidLocations[MSR_Counter, 0]
                        lon = ERA5Locations_near_MSR_CentroidLocations[MSR_Counter, 1]
                        index_lat = np.where(np_lat == lat)[0][0]
                        index_lon = np.where(np_lon == lon)[0][0]

                        u=np_ERA5Data_Inst.variables["u100"][:,index_lat, index_lon]
                        v=np_ERA5Data_Inst.variables["v100"][:,index_lat, index_lon]

                        speed = np.sqrt(u*u + v*v)
                        speed = np.atleast_1d(speed)

                        rows.append(speed[None, :])
                        
                    pd_allMSR_Hourly100mEffectiveWindSpeeds=pd.DataFrame(np.vstack(rows))
                    
                    stats_df = pd.DataFrame(dc_ResourceStatsAcrossMSR)
                    pd_Wind_GWA_MSR_Mean = stats_df["mean"].fillna(stats_df["mean"].mean()) # double checked, as long the MSR ids are numbered 0,1,2..., this command puts right mean to right table cell
                    # Get corrected 100m wind speeds, rows=MSRs, columns=hours

                    if year == ERA5Data_fityear:
                        print(f"Fit year {year}: Calculate bias-corrected time series and fit parameters")
                        S0_fityear, Shat0_fityear, pd_allMSR_Hourly100m8760BiasCorrEffectiveWindSpeeds=develop_allMSR_8760BiasCorrEffectiveWindSpeeds_fityear(pd_allMSR_Hourly100mEffectiveWindSpeeds, pd_Wind_GWA_MSR_Mean, TimeSteps)
                    
                    else:
                        if S0_fityear is None or Shat0_fityear is None:
                            raise RuntimeError("Fit year bias-correction not available.")
                        
                        print(f"Non-fit year {year}: Apply bias-correction using fit parameters from fit year {ERA5Data_fityear}")
                        pd_allMSR_Hourly100m8760BiasCorrEffectiveWindSpeeds=develop_allMSR_8760BiasCorrEffectiveWindSpeeds_year(pd_allMSR_Hourly100mEffectiveWindSpeeds, S0_fityear, Shat0_fityear, TimeSteps)

                for MSR_Counter in range(0,len(gpd_MSR_Attributes)): #MSR wise loop
                    lat=ERA5Locations_near_MSR_CentroidLocations[MSR_Counter,0]
                    lon=ERA5Locations_near_MSR_CentroidLocations[MSR_Counter,1]
                    index_lat = np.where(np_lat == lat)[0][0]
                    index_lon = np.where(np_lon == lon)[0][0]
                    np_Hourly2meterTemperature = np_ERA5Data_Inst.variables["t2m"][:, index_lat, index_lon]

                    if RE==SolarPVNameConvention:
                        np_HourlyGHI = np_ERA5Data_Acc.variables["ssrd"][:, index_lat, index_lon]
                        GSA_GHI_MSR_Mean=dc_ResourceStatsAcrossMSR[MSR_Counter]['mean']
                        if not GSA_GHI_MSR_Mean:
                            GSA_GHI_MSR_Mean=np.sum(np_HourlyGHI)*(24/TimeSteps)

                        np_HourlyCF, GHI_corrected_wh, BiasCorrection_GHI_Adder_Wh, ERA5_GHI_OriginalAnnualYield=ComputeHourlyCF_SolarPV(GSA_GHI_MSR_Mean,np_Hourly2meterTemperature,np_HourlyGHI)

                        np_HourlyCF=np.round(np_HourlyCF,decimals=3)
                        np_HourlyGHI = np.round(np_HourlyGHI/3600, decimals=3) #display in Watt Hours because the GHI biascorrect adder is also in Wh
                        np_HourlyGHI_corrected_Wh = np.round(GHI_corrected_wh, decimals=3) # In Watt Hours because the GHI biascorrect adder is also in Wh

                        np_allMSR_HourlyCF[MSR_Counter]=np_HourlyCF
                        np_allMSR_HourlyGHI[MSR_Counter] = np_HourlyGHI
                        np_allMSR_HourlyGHI_corrected_Wh[MSR_Counter]=np_HourlyGHI_corrected_Wh
                        np_allMSR_GHI_ERA5OriginalAnnualYield[MSR_Counter] = ERA5_GHI_OriginalAnnualYield
                        np_allMSR_GHI_GSA_MSR_Mean[MSR_Counter]= GSA_GHI_MSR_Mean*(TimeSteps/24) #converting per day yield of Solar GIS to annual
                        np_allMSR_BiasCorrection_GHI_Adder_Wh[MSR_Counter]=BiasCorrection_GHI_Adder_Wh

                    if RE==WindNameConvention:
                        np_HourlyU100mEastward=np_ERA5Data_Inst.variables["u100"][:,index_lat, index_lon]
                        np_HourlyV100mNorthward = np_ERA5Data_Inst.variables["v100"][:, index_lat, index_lon]
                        np_HourlyU10mEastward = np_ERA5Data_Inst.variables["u10"][:, index_lat, index_lon]
                        np_HourlyV10mNorthward = np_ERA5Data_Inst.variables["v10"][:, index_lat, index_lon]
                        Wind_GWA_MSR_Mean=dc_ResourceStatsAcrossMSR[MSR_Counter]['mean']
                        BiasCorrEffectiveWindSpeeds= pd_allMSR_Hourly100m8760BiasCorrEffectiveWindSpeeds.iloc[MSR_Counter, :]
                        if not Wind_GWA_MSR_Mean:
                            Wind_GWA_MSR_Mean=pd.DataFrame(dc_ResourceStatsAcrossMSR)['mean'].mean()
                        elevation=np_ERA5Data_Inst.variables["z"][0, index_lat, index_lon]/9.80665 #see orography variable description in https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

                        np_HourlyCF, np_HourlyWindSpeeds_ResultantVector, np_HourlyWindSpeeds_UnCorrected_ResultantVector, ERA5_AnnualMean=ComputeHourlyCF_Wind(BiasCorrEffectiveWindSpeeds, np_HourlyU100mEastward, np_HourlyU10mEastward, np_HourlyV100mNorthward, np_HourlyV10mNorthward, WindTurbineHeight_meters, np_Hourly2meterTemperature, elevation, pd_WindSpeed_to_Power)


                        np_HourlyCF = np.round(np_HourlyCF, decimals=3)
                        np_HourlyWindSpeeds_ResultantVector = np.round(np_HourlyWindSpeeds_ResultantVector, decimals=3)
                        np_HourlyWindSpeeds_UnCorrected_ResultantVector = np.round(np_HourlyWindSpeeds_UnCorrected_ResultantVector, decimals=3)

                        np_allMSR_HourlyCF[MSR_Counter] = np_HourlyCF
                        np_allMSR_HourlyWindSpeeds_corrected_ResultantVector[MSR_Counter] = np_HourlyWindSpeeds_ResultantVector
                        np_allMSR_HourlyWindSpeeds_uncorrected_ResultantVector[MSR_Counter] = np_HourlyWindSpeeds_UnCorrected_ResultantVector
                        np_allMSR_ERA5_AnnualMean[MSR_Counter] = ERA5_AnnualMean
                        np_allMSR_Wind_GWA_MSR_Mean[MSR_Counter]= Wind_GWA_MSR_Mean
                    print(f"{MSR_Counter} MSR {RE}")

                # Prepare the data set and export to xlsx
                pd_output = gpd_MSR_Attributes
                pd_output.rename(columns={"FID": "MSR_ID"}, inplace=True)
                HourTags = [f'H{i}' for i in range(1, TimeSteps+1)]
                if RE==SolarPVNameConvention:
                    print("creating solarpv files")
                    pd_SolarBiasInformation = pd.DataFrame(
                        {"ERA_GHI KWh/m2/yr": np_allMSR_GHI_ERA5OriginalAnnualYield, "GSA_GHI KWh/m2/yr": np_allMSR_GHI_GSA_MSR_Mean,
                        'BiasCorrection Adder Wh for solar hours': np_allMSR_BiasCorrection_GHI_Adder_Wh})
                    if flag_Diagnosis:
                        pd_output_diagnosis = pd.concat(
                            [pd_output,
                            pd_SolarBiasInformation,
                            pd.DataFrame(np_allMSR_HourlyGHI_corrected_Wh,
                                        columns=HourTags)],
                                        axis=1)
                        output_path = os.path.join(OutputCountryFolder_UTC, f"{country} {RE} BiasCorrected ResourceProfiles.csv")
                        pd_output_diagnosis.to_csv(output_path, index=False, sep=";")
                        print(f"{country} {RE} BiasCorrected ResourceProfiles.csv created")

                    pd_output = pd.concat(
                        [pd_output,
                        pd_SolarBiasInformation,
                        pd.DataFrame(np_allMSR_HourlyCF, columns=HourTags)], axis=1)
                    FileAddressCountryProfile_UTC = os.path.join(OutputCountryFolder_UTC, f"{country} {RE} {year} CFs.csv")
                    pd_output.to_csv(FileAddressCountryProfile_UTC, index=False, sep=";")
                    print(f"UTC-->{country} {RE} {year} CFs.csv created")

                if RE==WindNameConvention:
                    print("creating wind files")
                    pd_WindBiasInformation = pd.DataFrame(
                        {"GWA Annual MSR Mean m/s": np_allMSR_Wind_GWA_MSR_Mean,
                        "ERA-Raw Annual Mean Speed m/s":np_allMSR_ERA5_AnnualMean})
                    
                    if flag_Diagnosis:
                        pd_output_diagnosis = pd.concat(
                            [pd_output,
                            pd_WindBiasInformation,
                            pd.DataFrame(np_allMSR_HourlyWindSpeeds_corrected_ResultantVector,
                                        columns=HourTags)],
                                        axis=1)
                        output_path = os.path.join(OutputCountryFolder_UTC, f"{country} {RE} {WindTurbineHeight_meters}m BiasCorrected ResourceProfiles.csv")
                        pd_output_diagnosis.to_csv(output_path, index=False, sep=";")
                        print(f"{country} {RE} corrected resource profiles.csv created")

                        pd_output_diagnosis = pd.concat(
                            [pd_output, pd_WindBiasInformation, pd.DataFrame(np_allMSR_HourlyWindSpeeds_uncorrected_ResultantVector, columns=HourTags)], axis=1)
                        output_path = os.path.join(OutputCountryFolder_UTC, f"{country} {RE} {WindTurbineHeight_meters}m UnCorrected ResourceProfiles.csv")
                        pd_output_diagnosis.to_csv(output_path, index=False, sep=";")
                        print(f"{country} {RE} uncorrected resource profiles.csv created")

                    pd_output = pd.concat(
                        [pd_output,
                        pd_WindBiasInformation,
                        pd.DataFrame(np_allMSR_HourlyCF,
                                    columns=HourTags)],
                                    axis=1)
                    FileAddressCountryProfile_UTC = os.path.join(OutputCountryFolder_UTC,f"{country} {RE} {WindTurbineHeight_meters}m {year} CFs.csv")
                    pd_output.to_csv(FileAddressCountryProfile_UTC, index=False, sep=";")
                    print(f"UTC-->{country} {RE} {year} CFs.csv created")

                pd_UTC = pd.read_csv(FileAddressCountryProfile_UTC, sep=";")

                pd_LocalTime = CreateLocalTimeProfile(pd_UTC, pd_CountryUTC_offsets, country_withspaces)
                output_path = os.path.join(OutputCountryFolder_LocalTime, os.path.basename(FileAddressCountryProfile_UTC))
                pd_LocalTime.to_csv(output_path, index=False, sep=";")
                print(f"LocalTime-->{os.path.basename(FileAddressCountryProfile_UTC)}")

            except Exception as e:
                print (e)
                traceback.print_exc()
                pd_LogFile = pd.concat(
                    [pd_LogFile, 
                    pd.DataFrame({"Log": [f"{country}: Skipped {RE} profiles {year}"]})],
                    ignore_index=True)
                print(f"{country}: Skipped {RE} profiles {year}")
                output_path = os.path.join(OutputFolder_UTCProfiles, f"{DateTimeStamp}_{year}_ProfileGenerator_LogFile.csv")
                pd_LogFile.to_csv(output_path, index=False, sep=";")