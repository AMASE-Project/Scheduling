#%%
import setting
import simulation_utils as sim_utils
import simulation as sim

sim_utils.setup_iers_offline()
#%%
if __name__ == "__main__":
    observer_location = setting.observer_location
    data_file = '/home/chengz/Code/Scheduling/data/targets/combined_sample_v2.fits'
    time_resolution = setting.time_resolution
    start_date = '2026-01-01'
    N_days = 365
    sim_2026 = sim.ScheduleSimulation(observer_location, data_file,
                             time_resolution, start_date, N_days)
    sim_2026.load_data()
    outfile = '/home/chengz/Code/Scheduling/output/simulation_schedule.txt'
    sim_2026.run_simulation_multi(start_date=start_date, N_days=N_days, outfile=outfile)
    sim_2026.plot_sky_distribution()


# %%
