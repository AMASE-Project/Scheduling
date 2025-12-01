# %%
import setting
import simulation_utils as sim_utils

from tqdm.auto import tqdm
import logging
import os
import sys
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
#%%
class ScheduleSimulation():
    """A class to simulate the scheduling of astronomical observations over multiple nights."""
    def __init__(self, observer_location, data_file,
                 time_resolution, start_date, N_days,
                 weather_factor=setting.weather_factor, overhead_time=setting.overhead_time,
                 altitude_limit=setting.altitude_limit, moon_separation_limit=setting.moon_separation_limit,
                 moon_illumination_limit=setting.moon_illumination_limit, twilight_limit=setting.twilight_limit,
                 alt_w = setting.alt_w, moon_w = setting.moon_w, basic_w = setting.basic_w,
                 dic = '/home/chengz/Code/Scheduling/', # please set your own path here
                 bs_file = '/data/score/basic_score/bsm.npy',
                 as_file = '/data/score/altitude_score/asm.npy' 
                 ):
        """
        Initialize the ScheduleSimulation class.

        """
        self.observer_location = observer_location
        self.data_file = data_file
        self.time_resolution = time_resolution
        self.start_date = start_date
        self.N_days = N_days
        self.weather_factor = weather_factor
        self.altitude_limit = altitude_limit
        self.moon_separation_limit = moon_separation_limit
        self.moon_illumination_limit = moon_illumination_limit
        self.twilight_limit = twilight_limit
        self.overhead_time = overhead_time
        self.alt_w = alt_w
        self.moon_w = moon_w
        self.basic_w = basic_w
        self.dic = dic
        self.bs_file = bs_file
        self.as_file = as_file
        

        self.observer = Observer(location=self.observer_location, name="Nashan Observatory")
        self.lst_grid = np.arange(0,24, self.time_resolution.to(u.hour).value) * u.hourangle
        self.year = self.start_date[0:4]

        self.bsm = None
        self.asm = None
        self.night_windows = None
        self.targets = []
        self.schedule = {}
    
    def check_observed_state(self):
        """ 
        Check the observed state of all targets.
        ----------------------------------------------------------------------------
        Returns:
        observed_state : np.array
            An array indicating whether each target has been observed.
        """
        self.observed_state = np.array([target.observed for target in self.targets])
        return self.observed_state
    
    def load_night_windows(self,date_str):
        """Load or generate night windows for a given date.
        ----------------------------------------------------------------------------
        Parameters:
        date_str : str
            Date string in 'YYYY-MM-DD' format.
        Returns:
        night_windows : dict
            Dictionary containing night window information.
        """
        year_string = date_str[0:4]
        night_windows_file = self.dic + 'data/score/moon_score/' + year_string + '/night_window_multi.pkl'
        if os.path.exists(night_windows_file):
            with open(night_windows_file, 'rb') as f:
                night_windows = pickle.load(f)
        else:
            night_windows = sim_utils.generate_night_window_multi(self.start_date, 
                                                                  self.N_days, 
                                                                  self.observer,
                                                                  self.time_resolution,
                                                                  self.twilight_limit,
                                                                  save_path=night_windows_file)
        return night_windows
    
    def load_bsm(self):
        """Load or generate basic score matrix (BSM).
        ----------------------------------------------------------------------------
        Returns:
        bsm : np.array
            Basic score matrix.
        """
        bsm_file = self.dic + self.bs_file
        if os.path.exists(bsm_file):
            bsm = np.load(bsm_file)
        else:
            bsm = sim_utils.basic_score_matrix(self.targets, self.time_resolution, save_path=bsm_file)
        return bsm
    
    def load_asm(self):
        """Load or generate altitude score matrix (ASM).
        ----------------------------------------------------------------------------
        Returns:
        asm_norm : np.array
            Normalized altitude score matrix."""
        asm_file = self.dic + self.as_file
        if os.path.exists(asm_file):
            asm = np.load(asm_file)
            asm_norm = asm / np.max(asm, axis=1, keepdims=True)
        else:
            asm = sim_utils.altitude_score_matrix(self.targets, 
                                                  self.time_resolution, 
                                                  self.observer, 
                                                  self.altitude_limit, 
                                                  save_path=asm_file)
            asm_norm = asm / np.max(asm, axis=1, keepdims=True)
        return asm_norm
    
    def load_msm(self, date_str):
        """Load moon score matrix (MSM) for a given date.
        ----------------------------------------------------------------------------
        Parameters:
        date_str : str
            Date string in 'YYYY-MM-DD' format.
        Returns:
        msm : np.array
            Moon score matrix.
        """
        year_string = date_str[0:4]
        msm_file = self.dic + 'data/score/moon_score/' + year_string + '/matrix/msm_' + date_str + '.npy'
        if os.path.exists(msm_file):
            msm = np.load(msm_file)
        else:
            print('Please pre-calculate the moon score matrix first! Using pre_cal_moon.py')
            msm = None
        return msm

    def load_tsm(self, date_str, mask = None):
        """Load or generate total score matrix (TSM) for a given date.
        ----------------------------------------------------------------------------
        Parameters:
        date_str : str
            Date string in 'YYYY-MM-DD' format.
        mask : np.array, optional
            Optional mask to apply to the score matrix.
        Returns:
        tsm : np.array
            Total score matrix.
        """
        asm = self.load_asm()
        bsm = self.load_bsm()
        msm = self.load_msm(date_str)
        if msm is None:
            print('Cannot load moon score matrix, TSM cannot be calculated.')
            return None
        tsm = (self.alt_w * asm + self.moon_w * msm + self.basic_w * bsm) / (self.alt_w + self.moon_w + self.basic_w)
        valid_index = (asm>0) & (bsm>0) & (msm>0)
        if mask is not None:
            valid_index = valid_index & mask
        tsm[~valid_index] = 0
        return tsm
    
    def load_data(self):
        """Load target data from the specified data file."""
        data = fits.open(self.data_file)[1].data
        for row in tqdm(data):
            name = row['OBJID']
            ra = row['RA']
            dec = row['DEC']
            exptime = row['EXPTIME'] * u.s
            num_exp = row['NUMEXP']
            dither = row['DITHER']
            priority = row['PRIORITY']
            source = row['Source']
            if exptime>0*u.s and num_exp>0 and dither>0:
                max_altitude = 90 - abs(dec - self.observer_location.lat.value)
                if max_altitude < self.altitude_limit.value:
                    continue # Skip targets that never rise above altitude limit
                target = FixedTarget(name=name, coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs'))
                ori_time = row['EXPTIME'] * row['NUMEXP'] * row['DITHER']
                if ori_time < 3 * 60:
                    ori_time = 3 * 60  # Minimum observation time is 3 minutes
                elif ori_time < 15 * 60:
                    ori_time = 15 * 60  # Minimum observation time is 15 minutes
                elif ori_time < 30 * 60:
                    ori_time = 30 * 60  # Minimum observation time is 30 minutes
                else:
                    mm = ori_time // (1800) + 1
                    ori_time = mm * 1800  # Round up to the next half hour
                target.exposure_time = TimeDelta(ori_time * u.second + self.overhead_time)
                target.priority = int(priority)
                target.observed = False
                target.source = source
                target.state = 'Unobserved'
                self.targets.append(target)
        print(f"✓ Loaded {len(self.targets)} valid targets from data file. Skipped targets that never rise above altitude limit.")
        self.bsm = self.load_bsm()
        print("✓ Loaded basic score matrix.")
        self.asm = self.load_asm()
        print("✓ Loaded altitude score matrix.")
    
    def get_night_window(self, date_str):
        """Get the night window indices for a given date.
        ----------------------------------------------------------------------------
        Parameters:
        date_str : str
            Date string in 'YYYY-MM-DD' format.
        Returns:
        night_window_idx : np.array
            Indices of the night window in the LST grid.
        """
        self.night_windows = self.load_night_windows(date_str)
        print(f"✓ Loaded night windows for {date_str}.")
        if date_str not in self.night_windows:
            print(f"Night window for {date_str} not found!")
            return None
        (lst_sun_set, sun_set_time, lst_sun_rise, sun_rise_time, duration) = self.night_windows[date_str]
        if lst_sun_set is None:
            print(f"✗ No night time for date {date_str}.")
            return None
        if lst_sun_set < lst_sun_rise:
            night_window_idx = np.where((self.lst_grid >= lst_sun_set) & (self.lst_grid <= lst_sun_rise))[0]
        else:
            night_window_idx = np.where((self.lst_grid >= lst_sun_set) | (self.lst_grid <= lst_sun_rise))[0]
        return night_window_idx
    
    def run_simulation_night(self, date_str, mask=None):
        """Run the scheduling simulation for a single night.
        ----------------------------------------------------------------------------
        Parameters:
        date_str : str
            Date string in 'YYYY-MM-DD' format.
        mask : np.array, optional
            Optional mask to apply to the observation log.
        Returns:
        schedule : dict
            Schedule for the night.
        """
        print(f"★ Simulating observation for night {date_str}...")
        night_window_idx = self.get_night_window(date_str)
        schedule = {}

        # check if date_str is in night_windows
        if (date_str not in self.night_windows) or (self.night_windows is None):
            print(f"✗ Date {date_str} not in night windows.")
            schedule[date_str] = []
            return schedule
        
        # check if all targets have been observed
        observation_log = self.check_observed_state()
        if mask is not None:
            observation_log = observation_log | mask
        if np.all(observation_log):
            print("✓ All targets have been observed. No further scheduling needed.")
            schedule[date_str] = []
            return schedule
        
         # check the weather condition for the night
        weather_random = np.random.rand()
        if weather_random > self.weather_factor:
            print("✗ Bad weather for the night. No observations can be made.")
            schedule[date_str] = []
            return schedule
        
        
        if night_window_idx is None or len(night_window_idx) == 0:
            print(f"✗ No valid night window for date {date_str}.")
            schedule[date_str] = []
            return schedule
        
        night_length = len(night_window_idx)

        AS = self.asm
        BS = self.bsm
        MS = self.load_msm(date_str)[:, night_window_idx]

        BS = np.mean(BS, axis=1)
        MS = np.nanmean(MS, axis=1)

        AS_normalized = AS / np.sum(AS, axis=1, keepdims=True)
        AS_tonight = AS_normalized[:, night_window_idx]

        # only keep .3f
        AS_tonight = np.round(AS_tonight, 3)

        while night_length>0:
            AS_sum  = np.round(np.sum(AS_tonight, axis=1)* (~observation_log),3) # zero score for observed targets
            AS_max = np.round(np.nanmax(AS_tonight, axis=1) * (~observation_log),3)

            AS_sum[AS_sum>1] = 1

            max_sum_AS = np.nanmax(AS_sum)
            max_sum_AS_index = np.where(AS_sum == max_sum_AS)[0]

            if max_sum_AS <=0.01 or len(max_sum_AS_index)==0:
                print("✗ No more observable targets for tonight.")
                break

            # if multiple targets have the same max_sum_AS, choose the one with highest (AS_max + BS + MS)
            if len(max_sum_AS_index)>1:
                combined_scores = (self.alt_w * AS_max[max_sum_AS_index] + self.basic_w * BS[max_sum_AS_index] + self.moon_w * MS[max_sum_AS_index]) / (self.alt_w + self.basic_w + self.moon_w)
                best_index_in_max = np.argmax(combined_scores)
                target_index = max_sum_AS_index[best_index_in_max]
            else:
                target_index = max_sum_AS_index[0]
            
            # check how many lst slots are needed for the target
            target = self.targets[target_index]
            exptime = int(target.exposure_time.to(u.min).value)
            slots_needed = int(np.ceil(exptime / self.time_resolution.to(u.min).value))

            # check the remaining slots for this target in the night window
            AS_seq = AS_tonight[target_index, :]
            remaining_slots_index = np.where(AS_seq>0)[0]
            remaining_slots = len(remaining_slots_index)

            if len(remaining_slots_index) == 0:
                print('some error: no remaining slots available for target')
                break
            if remaining_slots <= slots_needed:
                # use all remaining slots
                chosen_slots = remaining_slots_index
            else:
                # find the best position to fit the slots_needed
                # sort the remaining_slots_index based on the AS score and choose the high needed slots
                sorted_indices = remaining_slots_index[np.argsort(-AS_seq[remaining_slots_index])]
                chosen_slots = sorted_indices[:slots_needed]
            
            # mark the chosen slots as used
            for slot in chosen_slots:
                AS_tonight[:, slot] = 0
            night_length -= len(chosen_slots)
            # check if the target can be fully observed
            if len(chosen_slots) * self.time_resolution.to(u.min).value >= exptime:
                # mark the target as observed
                observation_log[target_index] = True
                self.targets[target_index].observed = True
                self.targets[target_index].state = 'Observed'
                self.targets[target_index].exposure_time = TimeDelta(0 * u.second)
                # record the scheduled time
                scheduled_times = []
                for slot in chosen_slots:
                    lst_time = self.lst_grid[night_window_idx[slot]]
                    scheduled_times.append(round(lst_time.value, 2))
                # sort the scheduled times
                scheduled_times.sort()
                schedule.setdefault(date_str, []).append((self.targets[target_index], scheduled_times))
                print(f"✓ Scheduled target {self.targets[target_index].name} for observation on {date_str}.")
                print(f"The scheduled LST times are: {scheduled_times}.")
            else:
                self.targets[target_index].state = 'Partially Observed'
                self.targets[target_index].exposure_time -= TimeDelta((len(chosen_slots) * self.time_resolution.to(u.s).value) * u.second)
                if self.targets[target_index].exposure_time < TimeDelta(0.5 * u.min):
                    self.targets[target_index].exposure_time = TimeDelta(0 * u.second)
                    self.targets[target_index].observed = True
                    self.targets[target_index].state = 'Observed'
                    observation_log[target_index] = True
                
                scheduled_times = []
                for slot in chosen_slots:
                    lst_time = self.lst_grid[night_window_idx[slot]]
                    scheduled_times.append(round(lst_time.value, 2))
                # sort the scheduled times
                scheduled_times.sort()
                schedule.setdefault(date_str, []).append((self.targets[target_index], scheduled_times))
                print(f"✓ Partially scheduled target {self.targets[target_index].name} for observation on {date_str}.")
                print(f"The scheduled LST times are: {scheduled_times}.")
                print(f"Remaining needed exposure time: {round(self.targets[target_index].exposure_time.to(u.min).value, 1)} minutes.")
            print(f"Remaining night length: {night_length} blocks.")
        # summarize the night's scheduling
        if date_str in schedule:
            num_scheduled_targets = len(schedule[date_str])
            print(f"★ Finished scheduling for night {date_str}. Total scheduled targets: {num_scheduled_targets}.")
        left_targets = np.sum(~observation_log)
        print(f"★ Remaining unobserved/unfinished targets: {left_targets}/{len(self.targets)}.")

        return schedule
    
    def run_simulation_multi(self, start_date, N_days, outfile,mask=None):
        """Run the scheduling simulation over multiple nights.
        ----------------------------------------------------------------------------
        Parameters:
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        N_days : int
            Number of nights to simulate.
        outfile : str
            Output file to save the schedule.
        mask : np.array, optional
            Optional mask to apply to the observation log.
        Returns:
        all_schedules : dict
            Combined schedule for all nights.
        """
        all_schedules = {}
        date_list = [ (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(N_days)]
        original_stdout = sys.stdout
        with open(outfile, 'w') as f:
            sys.stdout = f
            for date_str in tqdm(date_list, desc="Simulating multiple nights"):
                daily_schedule = self.run_simulation_night(date_str, mask)
                all_schedules.update(daily_schedule)
        sys.stdout = original_stdout
        self.schedule = all_schedules
        print(f"✓ Finished multi-night simulation. Schedule saved to {outfile}.")
        # save all_schedules to a pickle file
        with open(outfile.replace('.txt', '.pkl'), 'wb') as f:
            pickle.dump(all_schedules, f)
        
        # save current targets information to a pickle file
        with open(outfile.replace('.txt', '_targets.pkl'), 'wb') as f:
            pickle.dump(self.targets, f)
        return all_schedules
    
    def plot_sky_distribution(self):
        """Plot the sky distribution of targets based on their observed state."""
        observed_targets = [target for target in self.targets if (target.state == 'Observed')]
        unobserved_targets = [target for target in self.targets if (target.state == 'Unobserved')]
        partobserved_targets = [target for target in self.targets if (target.state == 'Partially Observed')]
        observed_ra = [target.coord.ra.deg for target in observed_targets]       
        observed_dec = [target.coord.dec.deg for target in observed_targets]
        unobserved_ra = [target.coord.ra.deg for target in unobserved_targets]
        unobserved_dec = [target.coord.dec.deg for target in unobserved_targets]
        partobserved_ra = [target.coord.ra.deg for target in partobserved_targets]
        partobserved_dec = [target.coord.dec.deg for target in partobserved_targets]
        SkyCoord_observed = SkyCoord(ra=observed_ra*u.deg, dec=observed_dec*u.deg, frame='icrs')
        SkyCoord_unobserved = SkyCoord(ra=unobserved_ra*u.deg, dec=unobserved_dec*u.deg, frame='icrs')
        SkyCoord_partobserved = SkyCoord(ra=partobserved_ra*u.deg, dec=partobserved_dec*u.deg, frame='icrs')
        fig = plt.figure(figsize=(10, 5),dpi=200)
        ax = fig.add_subplot(111, projection='mollweide')
        ax.scatter(SkyCoord_unobserved.ra.wrap_at(180*u.deg).radian, SkyCoord_unobserved.dec.radian, s=3, color='gray', label='Unobserved Targets', alpha=0.3)
        ax.scatter(SkyCoord_observed.ra.wrap_at(180*u.deg).radian, SkyCoord_observed.dec.radian, s=3, color='blue', label='Observed Targets', alpha=0.6)
        ax.scatter(SkyCoord_partobserved.ra.wrap_at(180*u.deg).radian, SkyCoord_partobserved.dec.radian, s=3, color='red', label='Partially Observed Targets', alpha=0.6)
        xticks = np.arange(-150, 180, 30)
        xtick_labels = [(x + 360) % 360 for x in xticks]
        ax.set_xticks(xticks * np.pi / 180)
        ax.set_xticklabels(xtick_labels)

        ax.set_xlabel('Right Ascension (degrees)')
        ax.set_ylabel('Declination (degrees)')
        ax.set_title(f'Targets State', fontsize=16)
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.show()
        plt.close()

    
# %% test
"""
if __name__ == "__main__":
    observer_location = setting.observer_location
    data_file = '/home/chengz/Code/Scheduling/data/targets/combined_sample_v2.fits'
    time_resolution = setting.time_resolution
    start_date = '2026-01-01'
    N_days = 365
    sim = ScheduleSimulation(observer_location, data_file,
                             time_resolution, start_date, N_days)
    sim.load_data()
    outfile = '/home/chengz/Code/Scheduling/output/simulation_schedule.txt'
    sim.run_simulation_multi(start_date=start_date, N_days=N_days, outfile=outfile)
    sim.plot_sky_distribution()
"""