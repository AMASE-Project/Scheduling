#%%
import setting
import simulation_utils as sim_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pickle
from tqdm.auto import tqdm
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


sim_utils.setup_iers_offline()

#%%
"""
This script calculates the moon score matrix for a list of targets over a range of LST times and dates.
The moon score is based on the moon's illumination and its separation from the target.
This process is very time-consuming, so parallel processing is used to speed up the calculations.
And before the simulation, it's strong recommended to pre-calculate the moon score matrix by this script and save the results to disk.
Key Functions:
- moon_score: Calculate the moon score for a single target at a specific LST time.
- moon_score_target: Helper function to calculate moon scores for a single target across all LST grid points.
- moon_score_matrix: Calculate the moon score matrix for all targets over a range of LST
And the moon data should be pre-calculated and saved to disk using the `moon_data_multi` function from `simulation_utils.py`.
The night window data should also be pre-calculated and saved to disk using the `generate_night_window_multi` function from `simulation_utils.py`.

"""
#%%
moon_data_path = '/home/chengz/Code/Scheduling/data/score/moon_score/2026/moon_data_multi.pkl'
with open(moon_data_path, 'rb') as f:
    pre_moon_data =  pickle.load(f)

night_window_path = '/home/chengz/Code/Scheduling/data/score/moon_score/2026/night_window_multi.pkl'
with open(night_window_path, 'rb') as f:
    pre_night_window =  pickle.load(f)

#%%
pre_lst_grid = np.arange(0,24, setting.time_resolution.to(u.hour).value) * u.hourangle
pre_observer = Observer(location=setting.observer_location, name="Nashan Observatory")

#%%
def moon_score(target, observer, date_str, lst_time, moon_data, 
               moon_separation_limit=30 * u.deg, moon_illumination_limit=0.25):
    """
    Calculate the moon score of a target at a given LST time.
    Parameters:
    -----------
    target : astroplan.FixedTarget
        The target for which to calculate the moon score.
    observer : astroplan.Observer
        The observer location.
    date_str : str
        The date string in 'YYYY-MM-DD' format (e.g., '2024-01-01').
    lst_time : astropy.units.Quantity
        The Local Sidereal Time at which to calculate the moon score.
    moon_data : dict
        A dictionary with dates as keys and moon data dictionaries as values.
        The inner dictionary has LST times as keys and tuples containing:
        (observation_time, moon_altaz, moon_illumination) as values.
    moon_separation_limit : astropy.units.Quantity
        The minimum separation limit from the Moon (e.g., 30 * u.deg).
    moon_illumination_limit : float
        The maximum moon illumination fraction (e.g., 0.25).
    Returns:
    --------
    float
        The moon score of the target (0 to 1).
    """
    
    obs_time, moon_altaz, moon_illum = moon_data[date_str][lst_time.value]
    if obs_time is None:
        return 0
    # calculate the target's altaz at the obs_time
    target_altaz = observer.altaz(obs_time, target)
    moon_separation = abs(moon_altaz.separation(target_altaz).degree)
    if moon_illum < moon_illumination_limit:
        return 1.0  # full score for new moon
    else:
        if moon_separation > moon_separation_limit.value:
            return 1.0
        else:
            return moon_separation / moon_separation_limit.value

def moon_score_target(args):
    """
    Calculate moon scores for a single target across all LST grid points
    args: tuple, (i, target, date_str, moon_data, lst_grid, observer, moon_separation_limit, moon_illumination_limit)
    return: tuple, (i, moon_scores)
    """
    i, target, date_str = args
    moon_scores = np.zeros(len(pre_lst_grid))
    for j, lst_time in enumerate(pre_lst_grid):
        ms = moon_score(target, pre_observer, date_str, lst_time, pre_moon_data)
        moon_scores[j] = ms
    return i, moon_scores

def moon_score_matrix(targets, date_str,
                    use_parallel=True, max_workers=None):
    msm = np.zeros((len(targets), len(pre_lst_grid)))
    if use_parallel:
        if max_workers is None:
            max_workers = min(mp.cpu_count()-2, 16) # Leave some CPUs free
        # prepare args for each target
        args_list = []
        for i, target in enumerate(targets):
            args_list.append((
                i,
                target,
                date_str,
            ))
        # Use multiprocessing Pool for parallel calculation
        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(moon_score_target, args_list),
                                total=len(targets),
                                desc=f"Calculating moon scores in parallel with {max_workers} CPU for {date_str}"))
        # collect results
        for i, scores in results:
            msm[i, :] = scores
    else:
        for i, target in tqdm(enumerate(targets), total=len(targets), desc="Calculating moon scores sequentially"):
            for j, lst_time in enumerate(pre_lst_grid):
                ms = moon_score(target, pre_observer, date_str, lst_time, pre_moon_data)
                msm[i, j] = ms
    return msm
#%%
latitude = 43.4720 * u.deg

hdul = fits.open('/home/chengz/Code/Scheduling/data/targets/combined_sample_v2.fits')

data = hdul[1].data
targets = []

for row in data:
    name = row['OBJID']
    ra = row['RA']
    dec = row['DEC']
    exptime = row['EXPTIME'] * u.s
    num_exp = row['NUMEXP']
    dither = row['DITHER']
    priority = row['PRIORITY']
    if exptime>0*u.s and num_exp>0 and dither>0:
        max_altitude = 90 - abs(dec - latitude.value)
        if max_altitude < 30:
            continue  # skip targets that never rise above 30 degrees
        target = FixedTarget(name=name, coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs'))
        target.exposure_time = TimeDelta(row['EXPTIME'] * row['NUMEXP'] * row['DITHER'] * u.second)
        target.priority = int(priority)
        target.observed = False  
        targets.append(target)
hdul.close()
# %% calculate moon score matrix for a range of dates(e.g. 2026-01-01 to 2026-12-31)
for night_num in tqdm(range(200, 201), desc="Calculating moon score matrices for all dates in 2026"):
    date = datetime(2026, 1, 1) + timedelta(days=night_num - 1)
    date_str = date.strftime('%Y-%m-%d')
    msm = moon_score_matrix(targets, date_str,
                            use_parallel=True)
    # save the msm to disk
    save_dir = '/home/chengz/Code/Scheduling/data/score/moon_score/2026/matrix/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'msm_{date_str}.npy')
    np.save(save_path, msm)

# %%