import numpy as np
import pandas as pd
import xarray as xr
import gsw
import xscale.signal.fitting as xfit
from progressbar import ProgressBar

def preprocess_glider_profile(prof):
	"""
	Perform a pre-processing on the glider data

	"""
	# 1) Make a nice time record for the profile
	bottom = prof['P'].argmax(dim='NT')
	record = prof['time_since_start_of_dive']
	deltat_bottom = record.data[bottom].astype('f8')
	deltat_total = record.data[-1].astype('f8')
	alpha = deltat_bottom / deltat_total
	t_start = prof.GPS_time[0].data
	t_stop = prof.GPS_time[1].data
	t_bottom = t_start + pd.to_timedelta(
		alpha * (t_stop - t_start).astype('f8'))
	time_profile = t_bottom + pd.to_timedelta(
		prof['time_since_start_of_dive'] - deltat_bottom, unit='s')
	prof = prof.rename({'NT': 'time'}).assign_coords(
		time=('time', time_profile))

	# 2) Get the coordinates of the profile
	lat_start = prof.GPS_latitude[0].data
	lat_stop = prof.GPS_latitude[1].data
	lat_bottom = lat_start + alpha * (lat_stop - lat_start)
	lon_start = prof.GPS_longitude[0].data
	lon_stop = prof.GPS_longitude[1].data
	lon_bottom = lon_start + alpha * (lon_stop - lon_start)

	# 3) Clean up unvalid data
	mask = (prof['T'] > -2)
	niceT = prof['T'].where(mask, drop=True)
	niceS = prof['S'].where(mask, drop=True)
	# Do not forget to correct the offset due to surface pressure
	niceP = (prof['P'] - prof['Psurf']).where(mask, drop=True)
	niceDive = prof['dive'].where(mask, drop=True)

	# 4) Compute thermodynamic quantities from GSW toolbox
	# - Absolute Salinity
	SA = gsw.SA_from_SP(niceS, niceP, lat_start, lon_start)
	# - Conservative Temperature
	CT = gsw.CT_from_t(SA, niceT, niceP)
	# - In situ density
	rho = gsw.rho(SA, CT, niceP)
	# - Potential density referenced to surface pressure
	sigma0 = gsw.sigma0(SA, CT)
	# - Depth
	depth = - gsw.z_from_p(niceP, lat_start)

	# 5) Split the dive into one descending and one ascending path
	bottom = niceP.argmax(dim='time')
	ones = niceDive / niceDive
	newdive = xr.concat([2 * niceDive[:bottom] - 1, 2 * niceDive[bottom:]],
	                    dim='time')
	lat = xr.concat([0.5 * (lat_start + lat_bottom) * ones[:bottom],
	                 0.5 * (lat_stop + lat_bottom) * ones[bottom:]], dim='time')
	lon = xr.concat([0.5 * (lon_start + lon_bottom) * ones[:bottom],
	                 0.5 * (lon_stop + lon_bottom) * ones[bottom:]], dim='time')

	return xr.Dataset({'Theta': niceT, 'Salt': niceS, 'Pressure': niceP,
	                   'Rho': ('time', rho), 'Sigma0': ('time', sigma0)},
	                  coords={'profile': ('time', newdive.data),
	                          'depth': ('time', depth),
	                          'lat': ('time', lat), 'lon': ('time', lon)})


def preprocess_glider_multiprofiles(ds):
    list_of_profiles = []
    group_of_raw_data = ds.groupby('dive')
    pbar = ProgressBar(maxval=len(group_of_raw_data)).start()
    for i, ds in group_of_raw_data:
        list_of_profiles += [preprocess_glider_profile(ds.isel(NDIVES=(i-1)))]
        pbar.update(i)
    pbar.finish()
    return xr.concat(list_of_profiles, dim='time')


def interpolate_profile_on_density(prof, prof_number, density):
	"""
	Interpolate a profile with sparse measurement in time on a given density grid

	Parameters
	----------
	prof: xarray.Dataset
		Dataset containing the measured values for each time step
	density: 1darray
		Vector of in situ density on which the profile will be interpolated

	Returns
	-------
	new_profile: xarray.Dataset
		A profile with all the variables interpolated on the density grid
	"""
	from scipy.interpolate import interp1d

	# Compute fitting function from current profile
	ftime = interp1d(prof['Rho'], prof['time'], bounds_error=False)
	fTheta = interp1d(prof['Rho'], prof['Theta'], bounds_error=False)
	fSalt = interp1d(prof['Rho'], prof['Salt'], bounds_error=False)
	fdepth = interp1d(prof['Rho'], prof['depth'], bounds_error=False)

	# Regrid on a given density grid
	newtime = pd.to_datetime(ftime(density))
	newdepth = fdepth(density)
	newTheta = fTheta(density)
	newSalt = fSalt(density)
	newlat = prof['lat'].mean()
	newlon = prof['lon'].mean()
	new_profile = xr.Dataset(
		{'time': ('density', newtime), 'depth': ('density', newdepth),
		 'Theta': ('density', newTheta), 'Salt': ('density', newSalt)},
		coords={'density': ('density', density),
		        'profile': ('profile', [int(prof_number)]),
		        'lat': ('profile', [newlat]), 'lon': ('profile', [newlon])})
	return new_profile


def interpolate_multiprofile_on_density(ds, rho_step, rho_min=None,
                                        rho_max=None):
	"""
	Interpolate several profiles stored in a dataset on reference density levels

	Parameters
	----------
	prof: xarray.Dataset
		Dataset containing the different
	rho_step: float
		The step in density to construct the reference density levels
	rho_min: float, optional
		The lightest density value. If None, it is inferred from the dataset.
	rho_max: float, optional
		The heaviest density value. If None, it is inferred from the dataset.

	Returns
	-------
	res: xarray.Dataset
		A dataset of profiles with all the variables interpolated on the
		reference density levels
	"""
	if rho_min is None:
		rho_min = ds['Rho'].min()
	if rho_max is None:
		rho_max = ds['Rho'].max()
	density = np.arange(rho_min, rho_max, rho_step)
	list_of_new_profiles = []
	for i, prof in list(ds.groupby('profile')):
		list_of_new_profiles += [interpolate_profile_on_density(prof, i,
		                                                        density)]
	return xr.concat(list_of_new_profiles, dim='profile')


def fit_internal_tides(ds):
    """
    Fit M2 and K1 internal tides
    """
    ds = ds.dropna('profile').reset_coords(['lon', 'lat'])
    z = ds['depth'].rename({'profile': 'time'})
    z['time'].data = ds['time'].data
    new_profile = []
    list_of_fit = []
    for t in range(z.sizes['time']):
        middle = pd.to_datetime(ds.time[t].data)
        start = pd.to_datetime(middle - pd.Timedelta('1.5 days'))
        stop = pd.to_datetime(middle + pd.Timedelta('1.5 days'))
        ds_sample = z.sel(time=slice(start, stop))
        fit = xfit.sinfit(ds_sample.fillna(0.).chunk(chunks={'time':None}),
                          dim='time',
                          periods=[23.9345, 12.4206],
                          unit='h')
        fit = fit.assign(stdev=ds_sample.std())
        try:
            list_of_fit += [fit.load()]
            new_profile += [ds['profile'][t]]
        except ValueError:
            pass
    ds_tides = xr.concat(list_of_fit, dim='time',
                                      coords='different').\
	           rename({'time': 'profile'}).assign_coords(profile=new_profile)
    return ds_tides


def compute_tides(ds):
    list_of_fit = []
    pbar = ProgressBar(maxval=ds.sizes['density']).start()
    j=0
    for i, z in list(ds.groupby('density')):
        #prof = prof.dropna('profile').reset_coords(['lon', 'lat'])
        #z = prof['depth'].rename({'profile': 'time'})
        #z['time'].data = prof['time'].data
        try:
            zfit = fit_internal_tides(z).load()
            list_of_fit += [zfit,]
        except ValueError:
	        pass
        j += 1
        pbar.update(j)
    tides_fit = xr.concat(list_of_fit, dim='density').\
	            assign_coords(lon=ds.lon, lat=ds.lat)
    pbar.finish()
    return tides_fit