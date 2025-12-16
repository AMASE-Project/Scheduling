#%%
import sys
import os
import sys
script_path = '/home/chengz/Code/Scheduling/script' # please set your own path here
if script_path not in sys.path:
    sys.path.insert(0, script_path)
import setting
import simulation_utils as sim_utils
import simulation as sim
import numpy as np
sim_utils.setup_iers_offline()
#%%
# ----------------------------------
# Main simulation for 2 years from 2026-01-01
# ----------------------------------
if __name__ == "__main__":
    observer_location = setting.observer_location
    data_file = '/home/chengz/Code/Scheduling/data/targets/combined_sample_v2.fits'
    time_resolution = setting.time_resolution
    start_date = '2026-01-01'
    N_days = 365*2
    sim_2026 = sim.ScheduleSimulation(observer_location, data_file,
                             time_resolution, start_date, N_days)
    sim_2026.load_data()
    sim_2026.cal_single_block_length()
    #sim_2026.run_simulation_night(start_date)
    
    
    targets = sim_2026.targets
    mask_dig = np.zeros(len(targets), dtype=bool)
    for i in range(len(targets)):
        if (targets[i].source == 'DIG_ZHANGWEI') or (targets[i].source == 'WR_LIANG'):
            mask_dig[i] = True
    outfile = '/home/chengz/Code/Scheduling/output/simulation_schedule.txt'
    pngfile = '/home/chengz/Code/Scheduling/output/plot/sky.png'
    
    # maskplot = True, plot=True, mask=mask_dig
    #sim_2026.run_simulation_multi(start_date=start_date, N_days=N_days, outfile=outfile, pngfile=pngfile, maskplot = True, plot=True, mask=mask_dig)
    
    # For no mask plotting
    sim_2026.run_simulation_multi(start_date=start_date, N_days=N_days, outfile=outfile, pngfile=pngfile, maskplot = False, plot=False)
    sim_2026.plot_sky_distribution()

# %%
