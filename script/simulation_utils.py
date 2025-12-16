#%%
import sys
import os
sys.path.append('/home/chengz/Code/Scheduling/script')
import setting

from tqdm.auto import tqdm
import logging
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from astroplan import Observer, FixedTarget
from astroplan import is_observable, observability_table
from astroplan.moon import moon_illumination
import warnings
warnings.filterwarnings('ignore')
from astropy.utils.iers import conf as iers_conf
from astropy.coordinates import EarthLocation
from astropy.io import fits
iers_conf.auto_download = False
iers_conf.auto_max_age = None

from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord, get_body, AltAz
from datetime import datetime, timedelta

import multiprocessing as mp 

# %% Initialization
# Setup IERS to offline mode or use local cache if available
# This prevents unnecessary network calls during simulations
def setup_iers_offline():
    """
    Set IERS to offline mode or use local cache if available
    """
    # Suppress IERS-related warnings
    warnings.filterwarnings('ignore', category=Warning, module='astropy.utils.iers')
    
    # Try to use local cache if available
    iers_conf.auto_download = False  # Disable auto download
    
    # Check if IERS data is available
    try:
        from astropy.utils.iers import IERS_Auto
        iers_table = IERS_Auto.open()
        print("✓ Using local IERS data cache")
        return True
    except Exception as e:
        print(f"Local IERS data not available: {e}")
        return False
#%% function of getting night window
def generate_night_window(date, observer, time_resolution, twilight_limit):
    """
    Generate a time grid for the night of the given date.
    Parameters:
    -----------
    date : astropy Time object
        The date for which to generate the night time grid.
    observer : astroplan.Observer
        The observer location.
    time_resolution : astropy.units.Quantity
        Time resolution for the grid (e.g., 5 * u.min).
    twilight_limit : astropy.units.Quantity
        Twilight limit for the night (e.g., -12 * u.deg).
    Returns:
    --------
    dict
        A dictionary with the date as key and a tuple containing:
        (LST_sun_set, sun_set_time, LST_sun_rise, sun_rise_time, duration)
    
    Remarks:
    --------
    - LST_sun_set and LST_sun_rise are the Local Sidereal Times of sunset and sunrise.
    - sun_set_time and sun_rise_time are the actual times of sunset and sunrise.
    - duration is the total duration of the night in hours.
    - If the sun does not set or rise on the given date, returns None for all values.
    - The astroplan function observer.sun_set_time and observer.sun_rise_time may not work correctly; 
        this function uses a manual approach to find sunset and sunrise times.
    """
    date_list = {}
    date_str = date.strftime('%Y-%m-%d')
    midnight = observer.midnight(date, which='next')
    sun_set_time = midnight
    while observer.is_night(sun_set_time, horizon=twilight_limit):
        sun_set_time = sun_set_time - TimeDelta(time_resolution)
    sun_rise_time = midnight
    while observer.is_night(sun_rise_time, horizon=twilight_limit):
        sun_rise_time = sun_rise_time + TimeDelta(time_resolution)
    # ensure sun_set_time is before sun_rise_time
    if sun_set_time > sun_rise_time:
        sun_rise_time = observer.sun_rise_time(date + TimeDelta(1*u.day), which='next', horizon=twilight_limit)
    if sun_set_time > sun_rise_time:
        date_list[date_str] = (None, None, None, None, None)
        return date_list
    # convert to LST time
    lst_sun_set = observer.local_sidereal_time(sun_set_time)
    lst_sun_rise = observer.local_sidereal_time(sun_rise_time)
    duration = (sun_rise_time - sun_set_time).to(u.hour)
    date_list[date_str] = (lst_sun_set, sun_set_time, lst_sun_rise, sun_rise_time, duration)
    return date_list

def generate_night_window_multi(start_date_str, N_days, observer, time_resolution, twilight_limit, save_path=None):
    """
    Generate night windows for multiple days.
    Parameters:
    -----------
    start_date_str : str
        The starting date string in 'YYYY-MM-DD' format (e.g., '2024-01-01').
    N_days : int
        Number of days to generate night windows for (e.g., 365).
    observer : astroplan.Observer
        The observer location.
    time_resolution : astropy.units.Quantity
        Time resolution for the grid (e.g., 5 * u.min).
    twilight_limit : astropy.units.Quantity
        Twilight limit for the night (e.g., -12 * u.deg).
    Returns:
    dict
        A dictionary with dates as keys and tuples containing:
        (LST_sun_set, sun_set_time, LST_sun_rise, sun_rise_time, duration) as values.
    """
    date_list = {}
    start_date = Time(start_date_str)
    for day_offset in tqdm(range(N_days), desc="Generating night windows"):
        current_date = start_date + TimeDelta(day_offset * u.day)
        date_str = current_date.strftime('%Y-%m-%d')
        night_window = generate_night_window(current_date, observer, time_resolution, twilight_limit)
        date_list[date_str] = night_window[date_str]
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        # save the night window data to .pkl file
        with open(save_path, 'wb') as f:
            pickle.dump(date_list, f)
        print(f"Night window data saved to {save_path}")
    return date_list

#%% function of calculating score
# Basic Score Matrix
def basic_score_matrix(targets, time_resolution, save_path=None):
    """
    Calculate the basic score matrix for the given targets.
    Parameters:
    -----------
    targets : list of astroplan.FixedTarget
        List of targets to calculate the score matrix for.
    time_resolution : astropy.units.Quantity
        Time resolution for the grid (e.g., 5 * u.min).
    Returns:
    --------
    np.ndarray
        A 2D numpy array representing the basic score matrix.
    """
    time_grid = np.arange(0, 24, time_resolution.to(u.hour).value)
    bsm = np.ones((len(targets), len(time_grid)))
    for i, target in enumerate(tqdm(targets, desc="Calculating basic score matrix")):
        bsm[i, :] = (4 - target.priority) / 3.0
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, bsm)
        print(f"Basic score matrix saved to {save_path}")
    return bsm
    
# Altitude Score Matrix
def altitude_angle(target, lst_time, observer):
    """
    Calculate the altitude angle of a target at a given LST time.
    Parameters:
    -----------
    target : astroplan.FixedTarget
        The target for which to calculate the altitude angle.
    lst_time : astropy.units.Quantity
        The Local Sidereal Time at which to calculate the altitude angle.
    observer : astroplan.Observer
        The observer location.
    Returns:
    --------
    astropy.units.Quantity
        The altitude angle of the target in degrees.
    """
    lst_angle = lst_time.to(u.deg)
    HA = lst_angle - target.coord.ra
    sinh = np.sin(observer.latitude.to(u.rad).value) * np.sin(target.coord.dec.to(u.rad).value) + \
        np.cos(observer.latitude.to(u.rad).value) * np.cos(target.coord.dec.to(u.rad).value) * np.cos(HA.to(u.rad).value)
    alt = np.arcsin(sinh) * u.rad
    alt = alt.to(u.deg)
    return alt

def altitude_score(target, lst_time, observer, altitude_limit):
    """
    Calculate the altitude score of a target at a given LST time.
    Parameters:
    -----------
    target : astroplan.FixedTarget
        The target for which to calculate the altitude score.
    lst_time : astropy.units.Quantity
        The Local Sidereal Time at which to calculate the altitude score.
    observer : astroplan.Observer
        The observer location.
    altitude_limit : astropy.units.Quantity
        The minimum altitude limit for observations (e.g., 30 * u.deg).
    Returns:
    --------
    float
        The altitude score of the target (0 to 1).
    """
    alt = altitude_angle(target, lst_time, observer)
    if alt < altitude_limit:
        return 0.0
    else:
        return (alt - altitude_limit) / (90*u.deg - altitude_limit)
def altitude_score_matrix(targets, time_resolution, observer, altitude_limit, save_path=None):
    """
    Calculate the altitude score matrix for the given targets.
    Parameters:
    -----------
    targets : list of astroplan.FixedTarget
        List of targets to calculate the altitude score matrix for.
    time_resolution : astropy.units.Quantity
        Time resolution for the grid (e.g., 5 * u.min).
    observer : astroplan.Observer
        The observer location.
    altitude_limit : astropy.units.Quantity
        The minimum altitude limit for observations (e.g., 30 * u.deg).
    Returns:
    --------
    np.ndarray
        A 2D numpy array representing the altitude score matrix.
    """
    time_grid = np.arange(0, 24, time_resolution.to(u.hour).value) * u.hourangle
    asm = np.zeros((len(targets), len(time_grid)))
    for i, target in enumerate(tqdm(targets, desc="Calculating altitude score matrix")):
        for j, lst in enumerate(time_grid):
            asm[i, j] = altitude_score(target, lst, observer, altitude_limit)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, asm)
        print(f"Altitude score matrix saved to {save_path}")
    return asm
# Moon Separation Score Matrix
def get_moon_data_night(night_window_grid, observer, date, time_resolution):
    """
    Get moon data for a specific night.
    Parameters:
    -----------
    night_window_grid : dict
        A dictionary with dates as keys and night window tuples as values.
    observer : astroplan.Observer
        The observer location.
    date : astropy Time object
        The date for which to get moon data.
    time_resolution : astropy.units.Quantity
        Time resolution for the grid (e.g., 5 * u.min).
    Returns:
    --------
    dict
        A dictionary with the date as key and another dictionary as value.
        The inner dictionary has LST times as keys and tuples containing:
        (observation_time, moon_altaz, moon_illumination) as values.
    """
    date_str = date.strftime('%Y-%m-%d')
    night_window = night_window_grid[date_str]
    if night_window[0] is None:
        print(f"✗ No night time for date {date_str}.")
        return None, None, None
    sun_set_lst, sun_set_time, sun_rise_lst, sun_rise_time, duration = night_window
    moon_data = {}
    moon_data[date_str] = {}
    lst_grid = np.arange(0,24, time_resolution.to(u.hour).value) * u.hourangle
    for lst_time in lst_grid:
        # check if lst_time is within the night time window
        if sun_set_lst < sun_rise_lst:
            if lst_time < sun_set_lst or lst_time > sun_rise_lst:
                moon_data[date_str][lst_time.value] = (None, None, None)
                continue
        else:
            if lst_time < sun_set_lst and lst_time > sun_rise_lst:
                moon_data[date_str][lst_time.value] = (None, None, None)
                continue
        # calculate the observation time corresponding to the lst_time
        if sun_set_lst < sun_rise_lst:
            delta_hourangle = (lst_time - sun_set_lst).to(u.hourangle).value
        else:
            if lst_time >= sun_set_lst:
                delta_hourangle = (lst_time - sun_set_lst).to(u.hourangle).value
            else:
                delta_hourangle = (lst_time + 24*u.hourangle - sun_set_lst).to(u.hourangle).value
        # convert delta_hourangle to time
        delta_time = TimeDelta(delta_hourangle * u.hour)
        obs_time = sun_set_time + delta_time * 0.99726956 # the ratio of sidereal time to solar time
        moon = get_body("moon", obs_time, location=observer.location)
        moon_altaz = moon.transform_to(AltAz(obstime=obs_time, location=observer.location))
        moon_illum = moon_illumination(obs_time)
        moon_data[date_str][lst_time.value] = (obs_time, moon_altaz, moon_illum)
    return moon_data

def moon_data_multi(night_window_grid, observer, start_date_str, N_days, time_resolution, 
                    use_parallel=True, max_workers=None, save_path=None):
    """
    Get moon data for multiple nights.
    Parameters:
    -----------
    night_window_grid : dict
        A dictionary with dates as keys and night window tuples as values.
    observer : astroplan.Observer
        The observer location.
    start_date_str : str
        The starting date string in 'YYYY-MM-DD' format (e.g., '2024-01-01').
    N_days : int
        Number of days to get moon data for (e.g., 365).
    time_resolution : astropy.units.Quantity
        Time resolution for the grid (e.g., 5 * u.min).
    use_parallel : bool, optional
        Whether to use parallel processing (default is True).
    max_workers : int or None, optional
        Number of parallel processes to use (default is None, which uses min(cpu_count()-2, 8)).
    Returns:
    --------
    dict
        A dictionary with dates as keys and moon data dictionaries as values.
        The inner dictionary has LST times as keys and tuples containing:
        (observation_time, moon_altaz, moon_illumination) as values.
    """
    star_date = Time(start_date_str)
    moon_data = {}
    if use_parallel:
        if max_workers is None:
            max_workers = min(mp.cpu_count()-2, 8) # Leave some CPUs free
        tasks = []
        for day_offset in range(N_days):
            current_date = star_date + TimeDelta(day_offset * u.day)
            tasks.append((
                night_window_grid,
                observer,
                current_date,
                time_resolution
            ))
        # Use multiprocessing Pool for parallel calculation
        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(
                pool.starmap(get_moon_data_night, tasks),
                total=len(tasks),
                desc="Calculating moon data, parallel ncpu = {}".format(max_workers)
            ))
        # Collect results into a single dictionary
        for day_offset, result in enumerate(results):
            current_date = star_date + TimeDelta(day_offset * u.day)
            date_str = current_date.strftime('%Y-%m-%d')
            moon_data[date_str] = result[date_str]
    else:
        # Sequential calculation
        for day_offset in tqdm(range(N_days), desc="Calculating moon data, sequential"):
            current_date = star_date + TimeDelta(day_offset * u.day)
            date_str = current_date.strftime('%Y-%m-%d')
            moon_data_night = get_moon_data_night(
                night_window_grid,
                observer,
                current_date,
                time_resolution
            )
            moon_data[date_str] = moon_data_night[date_str]
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        # save the moon data to .pkl file
        with open(save_path, 'wb') as f:
            pickle.dump(moon_data, f)
        print(f"Moon data saved to {save_path}")
    return moon_data


# %% test
"""
setup_iers_offline()

star_date_str = '2027-01-01'
N_days = 365
observer = Observer(location=setting.observer_location, name="Nashan Observatory")

night_window_multi = generate_night_window_multi(
    start_date_str=star_date_str,
    N_days=N_days,
    observer=observer,
    time_resolution=setting.time_resolution,
    twilight_limit=setting.twilight_limit,
    save_path='/home/chengz/Code/Scheduling/data/score/moon_score/'+star_date_str[:4]+'/night_window_multi.pkl'
)

mdm = moon_data_multi(
    night_window_grid=night_window_multi,
    observer=observer,
    start_date_str=star_date_str,
    N_days=N_days,
    time_resolution=setting.time_resolution,
    save_path='/home/chengz/Code/Scheduling/data/score/moon_score/'+star_date_str[:4]+'/moon_data_multi.pkl'
)
"""