#%% Setting.py
import astropy.units as u
from astropy.coordinates import EarthLocation

#%% General settings

# Define observer location (example: Nashan Observatory)
longitude = 87.1750 * u.deg
latitude = 43.4720 * u.deg
elevation = 2080 * u.m
location = EarthLocation.from_geodetic(longitude, latitude, elevation)
observer_location = location

# Define the time resolution for simulations
time_resolution = 5 * u.min  # 5 minutes

# Define the Constraints
altitude_limit= 30 * u.deg  # Minimum altitude for observations
twilight_limit = -12 * u.deg  # Twilight limit for observations
moon_separation_limit = 30 * u.deg # Minimum separation from the Moon
moon_illumination_limit = 0.25 # Maximum moon illumination fraction
