import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import xscale.signal.fitting as xfit
from progressbar import ProgressBar
from dask import delayed
from xarray.ufuncs import *

# Global variables
FLAG_DICT = {"Pass": 1, "Not Evaluated": 2, "Suspect or Of High Interest": 3,
			 "Fail": 4, "Missing data": 9}
FLAG_REVERSE_DICT = {1: "Pass", 2: "Not Evaluated",
					 3: "Suspect or Of High Interest", 4: "Fail",
					 9: "Missing data"}


class QcBatch(object):

	def __init__(self, raw_data, qctests=None):
		if qctests is not None:
			self.qctests = {test.get_name(): test for test in qctests}
		self.flag_dict = {}
		self.main_flag = None
		self.raw_data = raw_data
		self.profiles_data = None

#	def __add__(self, qctests):
#		self.qctests += qctests

	def __delitem__(self, key):
		del self.qctests[key]

	def __setitem__(self, key, value):
		self.qctests[key] = value

	def __repr__(self):
		message = ("Batch for Quality Control Tests \n"
				   "******************************* \n"
				   "Details by test: \n"
				   "---------------- \n")
		for test in self.qctests:
			message += self.qctests[test].__repr__()
			message += "\n"
		return message

	def set_qctests(self, qctests):
		self.qctests = {test.get_name(): test for test in qctests}

	def prepare_data(self):
		list_of_profiles = []
		group_of_raw_data = self.raw_data.groupby('dive')
		pbar = ProgressBar(maxval=len(group_of_raw_data)).start()
		for i, ds in group_of_raw_data:
			try:
				list_of_profiles += [
					compute_thermodynamics(ds.isel(NDIVES=(i-1)))]
			except ValueError:
				pass
			pbar.update()
		pbar.finish()
		self.profiles_data = xr.concat(list_of_profiles, dim='time')

	def run(self):
		if self.profiles_data is None:
			self.prepare_data()
		self.main_flag = xr.full_like(self.profiles_data['Temperature'], 2,
									  dtype='i8')
		self.main_flag.name = "Main flag"
		indx_fail = np.array([], dtype='i8')
		indx_suspect = np.array([], dtype='i8')
		indx_pass = np.array([], dtype='i8')
		for test in self.qctests:
			self.qctests[test].apply(self.profiles_data)
			flag = self.qctests[test].get_flag()
			self.flag_dict[test] = flag
			indx_fail = np.append(indx_fail, np.argwhere(flag.data == 4))
			indx_suspect = np.append(indx_suspect, np.argwhere(flag.data == 3))
			indx_pass = np.append(indx_pass, np.argwhere(flag.data == 1))

		self.main_flag.data[np.unique(indx_pass)] = 1
		self.main_flag.data[np.unique(indx_suspect)] = 3
		self.main_flag.data[np.unique(indx_fail)] = 4

	def plot(self, test_name=None, time_boundaries=None, **kwargs):
		if test_name is None:
			flag = self.main_flag
		else:
			flag = self.flag_dict[test_name]
		ax = plt.subplot(111)
		plt.scatter(flag.time.data, flag.depth.data, c=flag, cmap='Set1',
					vmin=1, vmax=9, **kwargs)
		if time_boundaries is None:
			ax.set_xlim([pd.to_datetime(flag.time.min().data),
						 pd.to_datetime(flag.time.max().data)])
		else:
			ax.set_xlim(time_boundaries)
		ax.invert_yaxis()
		plt.colorbar()

	def get_valid_data(self, suspect=True):
		if suspect:
			return self.profiles_data.where(self.main_flag != 4)
		else:
			return self.profiles_data.where((self.main_flag != 4) &
											(self.main_flag != 3))


class QcTest(object):

	def __init__(self):
		#self.data = data
		#self.nb_records = data.sizes['time']
		self.flag = None
		self.name = self.get_name()
		#self.flag = xr.full_like(self.data['Temperature'], 2, dtype='i8')
		#self.flag.name = 'Flag'
		#self.flag.attrs['summary'] = {"Pass": 0,
		#                              "Not Evaluated": 0,
		#                              "Suspect or Of High Interest": 0,
		#                              "Fail": self.nb_records,
		#                              "Missing data": 0}

	def __call__(self, data):
		self.apply(data)

	def __repr__(self):
		message = ("%s \n" % self.get_name()+
				   "----------------------------------------------------\n")
		if self.flag is None:
			message += "Test not run yet."
		else:
			nb_points = self.flag.sizes['time']
			for key in self.flag.attrs['summary'].keys():
				value = self.flag.attrs['summary'][key]
				percentage = value / nb_points * 100
				message += "%s: %s/%s (%s) \n" % (key, value, nb_points,
													percentage)
		return message

	def init_flag(self, data):
		self.flag = xr.full_like(data['Temperature'], 2, dtype='i8')
		self.flag.name = 'Flag'
		self.flag.attrs['summary'] = {"Pass": 0,
									  "Not Evaluated": 0,
									  "Suspect or Of High Interest": 0,
									  "Fail": data.sizes['time'],
									  "Missing data": 0}

	def set_flag(self, condition, value=9):
		# 1 = Pass
		# Data have passed critical real-time quality control tests
		# and are deemed adequate for use as preliminary data.
		#
		# 2 = Not evaluated, Data have not been QC-tested, or the information on
		# quality is not available.
		#
		# 3 = Suspect or Of High Interest
		# Data are considered to be either suspect or of high interest to data
		# providers and users.They are flagged suspect to draw further attention
		# to them by operators.
		#
		# 4 = Fail
		# Data are considered to have failed one or more critical real-time QC
		# checks.If they are disseminated at all, it should be readily apparent
		# that they are not of acceptable quality.
		#
		# 9 = Missing data
		# Data are missing; used as a placeholder
		self.flag.data[np.argwhere(condition.data)] = value
		self.flag.attrs['summary'][FLAG_REVERSE_DICT[value]] = \
		(self.flag == value).sum().data

	def get_flag(self):
		"""
		Get the flags returned by the current test
		"""
		return self.flag

	def get_name(self):
		return self.__class__.__name__

	def apply(self, data):
		pass

	def plot(self, time_boundaries=None, **kwargs):
		flag = self.flag
		ax = plt.subplot(111)
		plt.scatter(flag.time.data, flag.depth.data, c=flag, cmap='Set1',
					vmin=1, vmax=9, **kwargs)
		if time_boundaries is None:
			ax.set_xlim([pd.to_datetime(flag.time.min().data),
						 pd.to_datetime(flag.time.max().data)])
		else:
			ax.set_xlim(time_boundaries)
		ax.invert_yaxis()
		plt.colorbar()


class LocationTest(QcTest):

	def __init__(self, max_displacement=20):
		self.md = max_displacement
		QcTest.__init__(self)

	def apply(self, data):
		self.init_flag(data)
		lat = data['latitude']
		lon = data['longitude']
		distance = data['distance']
		impossible_location = ((lat > 90) | (abs(lat) > 90) |
							   (lon < 360) | (lon < 90))
		self.set_flag(impossible_location, value=4)

		unlikely_displacement = (distance < self.md)
		self.set_flag(unlikely_displacement, value=3)
		self.set_flag((1 - unlikely_displacement) *
					  (1 - impossible_location), value=1)


class GrossRangeTest(QcTest):

	def __init__(self, variable, user_min=None, user_max=None,
				 sensor_min=None, sensor_max=None):
		self.var = variable
		self.user_min = user_min
		self.user_max = user_max
		self.sensor_min = sensor_min
		self.sensor_max = sensor_max
		QcTest.__init__(self)



	def get_name(self):
		return self.__class__.__name__ + " on " + self.var

	def apply(self, data):
		self.init_flag(data)
		if self.user_min is None:
			self.user_min = data[self.var].min()
		else:
			self.user_min = self.user_min
		if self.user_max is None:
			self.user_max = data[self.var].max()
		else:
			self.user_max = self.user_max
		if self.sensor_min is None:
			self.sensor_min = data[self.var].min()
		else:
			self.sensor_min = self.sensor_min
		if self.sensor_max is None:
			self.sensor_max = data[self.var].max()
		else:
			self.sensor_max = self.sensor_max
		if not self.var in data.variables:
			raise ValueError('The name of the variable is incorrect')
		darray = data[self.var]
		outside_user_range = ((darray > self.user_max) |
							  (darray < self.user_min))
		self.set_flag(outside_user_range, value=3)
		outside_sensor_range = ((darray > self.sensor_max) |
								(darray < self.sensor_min))
		self.set_flag(outside_sensor_range, value=4)
		self.set_flag((1 - outside_sensor_range) * (1 - outside_user_range),
					  value=1)


class PressureTest(QcTest):

	def apply(self, data):
		self.init_flag(data)
		non_monotonic_pressure = ((data['Pressure'].diff('time') >=
								   0).all('time') |
								  (data['Pressure'].diff('time') <=
								   0).all('time')
								  )
		negative_pressure = data['Pressure'] < 0
		self.set_flag(negative_pressure, value=4)
		self.set_flag(non_monotonic_pressure, value=3)
		self.set_flag((1 - non_monotonic_pressure) *
					  (1 - negative_pressure), value=1)


class RateOfChangeTest(QcTest):

	def __init__(self, variable,  nb_stdev, nb_timestep):
		self.var = variable
		self.ndev = nb_stdev
		self.nt = nb_timestep
		QcTest.__init__(self)

	def get_name(self):
		return self.__class__.__name__ + " on " + self.var

	def apply(self, data):
		self.init_flag(data)
		if not self.var in data.variables:
			raise ValueError('The name of the variable is incorrect')
		darray = data[self.var]
		stdev = darray.rolling(time=self.nt, min_periods=1,
								  center=True).std()
		exceeding_rate = (abs(darray.diff('time')).fillna(0.) >
						  self.ndev * stdev)
		self.set_flag(exceeding_rate, value=3)
		self.set_flag(1 - exceeding_rate, value=1)


class DensityInversionTest(QcTest):

	def __init__(self, N2_min):
		self.dd = N2_min
		QcTest.__init__(self)

	def apply(self, data):
		self.init_flag(data)
		insufficient_increase = (data['N2'] <= self.dd)
		self.set_flag(insufficient_increase, value=4)
		self.set_flag(1 - insufficient_increase, value=1)



def compute_thermodynamics(prof):
	"""
	Perform a pre-processing on the glider data

	"""
	# 1) Make a nice time record for the profile
	max_depth = prof['P'].max(dim='NT')
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
	distance_start_to_bottom = 1e-3 * gsw.distance([lon_start, lon_bottom],
												  [lat_start, lat_bottom],
												  p=[0, max_depth]).squeeze()
	distance_bottom_to_stop = 1e-3 * gsw.distance([lon_bottom, lon_stop],
												  [lat_bottom, lat_stop],
												  p=[max_depth, 0]).squeeze()
	# 3) Clean up unvalid data
	niceT = prof['T']
	niceS = prof['S']
	# Do not forget to correct the offset due to surface pressure
	niceP = (prof['P'] - prof['Psurf'])
	niceDive = prof['dive']

	# 4) Compute thermodynamic quantities from GSW toolbox
	# - Absolute Salinity
	SA = gsw.SA_from_SP(niceS, niceP, lat_start, lon_start)
	# - Conservative Temperature
	CT = gsw.CT_from_t(SA, niceT, niceP)
	# - In situ density
	rho = gsw.rho(SA, CT, niceP)
	# - Potential density referenced to surface pressure
	sigma0 = gsw.sigma0(SA, CT)
	# - Buoyancy
	b = 9.81 * (1 - rho / 1025)
	N2 = xr.DataArray(gsw.Nsquared(SA, CT, niceP)[0], name='Buoyancy frequency',
					  dims='time',
					  coords = {'time': ('time',
										 prof['time'].isel(time=slice(1, None)))
								}
					  )
	# - Depth
	depth = - gsw.z_from_p(niceP, lat_start)

	# 5) Split the dive into one descending and one ascending path
	bottom = niceP.argmax(dim='time')
	ones = xr.ones_like(niceP)
	newdive = xr.concat([2 * niceDive[:bottom] - 1, 2 * niceDive[bottom:]],
						dim='time')
	lat = xr.concat([0.5 * (lat_start + lat_bottom) * ones[:bottom],
					 0.5 * (lat_stop + lat_bottom) * ones[bottom:]], dim='time')
	lon = xr.concat([0.5 * (lon_start + lon_bottom) * ones[:bottom],
					 0.5 * (lon_stop + lon_bottom) * ones[bottom:]], dim='time')
	Pmax = xr.concat([max_depth * ones[:bottom],
					  max_depth * ones[bottom:]], dim='time')

	distance = xr.concat([distance_start_to_bottom * ones[:bottom],
						  distance_bottom_to_stop * ones[bottom:]], dim='time')
	distance.name = 'distance between profiles in km'

	return xr.Dataset({'Temperature': niceT,
					   'Salinity': niceS,
					   'Pressure': niceP,
					   'Rho': ('time', rho),
					   'CT': ('time', CT),
					   'SA': ('time', SA),
					   'Sigma0': ('time', sigma0),
					   'b': ('time', b),
					   'Pmax': ('time', Pmax),
					   'N2': N2},
					  coords={'profile': ('time', newdive.data),
							  'depth': ('time', depth),
							  'lat': ('time', lat), 'lon': ('time', lon),
							  'distance': distance})


def preprocess_glider_multiprofiles(ds):
	list_of_profiles = []
	group_of_raw_data = ds.groupby('dive')
	pbar = ProgressBar(maxval=len(group_of_raw_data)).start()
	for i, ds in group_of_raw_data:
		try:
			list_of_profiles += [compute_thermodynamics(ds.isel(NDIVES=(i - 1)))]
		except ValueError:
			pass
		pbar.update(i)
	pbar.finish()
	return xr.concat(list_of_profiles, dim='time')


def interpolate_profile_on_density(prof, prof_number, density, potential=False):
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
	if potential:
		ftime = interp1d(prof['Sigma0'], prof['time'], bounds_error=False)
		fTheta = interp1d(prof['Sigma0'], prof['Temperature'],
						  bounds_error=False)
		fSalt = interp1d(prof['Sigma0'], prof['Salinity'], bounds_error=False)
		fdepth = interp1d(prof['Sigma0'], prof['depth'], bounds_error=False)
	else:
		ftime = interp1d(prof['Rho'], prof['time'], bounds_error=False)
		fTheta = interp1d(prof['Rho'], prof['Temperature'], bounds_error=False)
		fSalt = interp1d(prof['Rho'], prof['Salinity'], bounds_error=False)
		fdepth = interp1d(prof['Rho'], prof['depth'], bounds_error=False)

	# Regrid on a given density grid
	newtime = pd.to_datetime(ftime(density))
	newdepth = fdepth(density)
	newTheta = fTheta(density)
	newSalt = fSalt(density)
	newlat = prof['lat'].mean()
	newlon = prof['lon'].mean()
	newdist = prof['distance'].mean()
	new_profile = xr.Dataset(
		{'time': ('density', newtime), 'depth': ('density', newdepth),
		 'Temperature': ('density', newTheta), 'Salinity': ('density', newSalt)},
		coords={'density': ('density', density),
				'profile': ('profile', [int(prof_number)]),
				'lat': ('profile', [newlat]), 'lon': ('profile', [newlon]),
				'distance': ('profile', [newdist])})
	return new_profile


def interpolate_profile_on_depth(prof, prof_number, depth):
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
	ftime = interp1d(prof['depth'], prof['time'], bounds_error=False)
	fTheta = interp1d(prof['depth'], prof['Temperature'], bounds_error=False)
	fSalt = interp1d(prof['depth'], prof['Salinity'], bounds_error=False)
	fRho = interp1d(prof['depth'], prof['Rho'], bounds_error=False)

	# Regrid on a given density grid
	newtime = pd.to_datetime(ftime(depth))
	newRho = fRho(depth)
	newTheta = fTheta(depth)
	newSalt = fSalt(depth)
	newlat = prof['lat'].mean()
	newlon = prof['lon'].mean()
	newdist = prof['distance'].mean()
	new_profile = xr.Dataset(
		{'time': ('depth', newtime), 'Rho': ('depth', newRho),
		 'Temperature': ('depth', newTheta), 'Salinity': ('depth', newSalt)},
		coords={'depth': ('depth', depth),
				'profile': ('profile', [int(prof_number)]),
				'lat': ('profile', [newlat]), 'lon': ('profile', [newlon]),
				'distance': ('profile', [newdist])})
	return new_profile



def interpolate_multiprofile_on_density(ds, rho_step, rho_min=None,
										rho_max=None, potential=False):
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
																density,
																potential=potential)]
	return xr.concat(list_of_new_profiles, dim='profile')


def interpolate_multiprofile_on_depth(ds, z_step, z_min=None, z_max=None):
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
	if z_min is None:
		rho_min = ds['depth'].min()
	if z_max is None:
		rho_max = ds['depth'].max()
	depth = np.arange(z_min, z_max, z_step)
	list_of_new_profiles = []
	for i, prof in list(ds.groupby('profile')):
		list_of_new_profiles += [interpolate_profile_on_depth(prof, i, depth)]
	return xr.concat(list_of_new_profiles, dim='profile')


def fit_internal_tides(ds):
	"""
	Fit M2 and K1 internal tides
	"""
	ds = ds.dropna('profile').reset_coords(['distance', 'lon', 'lat'])
	z = ds['depth'].rename({'profile': 'time'})
	z['time'].data = ds['time'].data
	new_profile = []
	list_of_fit = []
	for t in range(z.sizes['time']):
		middle = pd.to_datetime(ds.time[t].data)
		start = pd.to_datetime(middle - pd.Timedelta('3 days'))
		stop = pd.to_datetime(middle + pd.Timedelta('3 days'))
		ds_sample = z.sel(time=slice(start, stop))
		fit = xfit.sinfit(ds_sample.fillna(0.).chunk(chunks={'time':None}),
						  dim='time',
						  periods=[23.9345, 12.4206],
						  unit='h')
		fit = fit.assign(stdev = ds_sample.std())
		try:
			list_of_fit.append(fit.load())
			new_profile.append(ds['profile'][t])
		except ValueError:
			pass
	ds_tides = xr.concat(list_of_fit, dim='time', coords='different')

	return ds_tides.rename({'time': 'profile'}).assign_coords(profile=new_profile)


def fit_internal_tides_2(ds):
	"""
	Fit M2 and K1 internal tides
	"""
	ds = ds.dropna('profile').reset_coords(['distance', 'lon', 'lat'])
	z = ds['depth'].rename({'profile': 'time'})
	z['time'].data = ds['time'].data
	new_profile = []
	list_of_fit = []
	for t in range(z.sizes['time']):
		middle = delayed(pd.to_datetime)(ds.time[t].data)
		start = delayed(pd.to_datetime)(middle - pd.Timedelta('1.5 days'))
		stop = delayed(pd.to_datetime)(middle + pd.Timedelta('1.5 days'))
		ds_sample = delayed(z).sel(time=slice(start, stop))
		fit = delayed(xfit.sinfit)(ds_sample.fillna(0.).chunk(chunks={'time':None}),
						  dim='time',
						  periods=[23.9345, 12.4206],
						  unit='h')
		stdev = delayed(ds_sample.std)()
		fit = delayed(fit.assign)(stdev = stdev)
		try:
			list_of_fit.append(fit)
			new_profile.append(ds['profile'][t])
		except ValueError:
			pass
	ds_tides = delayed(xr.concat)(list_of_fit, dim='time', coords='different')

	return ds_tides.rename({'time': 'profile'}).assign_coords(profile=new_profile)


def compute_tides(ds):
	list_of_fit = []
	pbar = ProgressBar(maxval=ds.sizes['density']).start()
	for i in range(ds.sizes['density']):
		#zfit = delayed(fit_internal_tides)(ds.isel(density=i))
		#list_of_fit.append(zfit)
		try:
			#zfit = delayed(fit_internal_tides)(ds.isel(density=i))
			zfit = fit_internal_tides(ds.isel(density=i))
			list_of_fit.append(zfit)
		except:
			pass
		pbar.update(i)
	#from dask.diagnostics import ProgressBar
	#with ProgressBar():
	#    tides_fit = delayed(xr.concat)(list_of_fit, dim='density').compute()
	#tides_fit = delayed(xr.concat)(list_of_fit, dim='density').compute()
	tides_fit = xr.concat(list_of_fit, dim = 'density')
	#tides_fit = xr.concat(res, dim='density')
	#pbar.register()
	#res = tides_fit.compute()
	#pbar.unregister()
	pbar.finish()
	return tides_fit.assign_coords(lon=ds.lon, lat=ds.lat)


def compute_tides_v2(ds):
	list_of_fit = []
	pbar = ProgressBar(maxval=ds.sizes['density']).start()
	for i in range(ds.sizes['density']):
		try:
			zfit = fit_internal_tides_2(ds.isel(density=i)).compute()
			list_of_fit.append(zfit)
		except:
			pass
		pbar.update(i)
	tides_fit = xr.concat(list_of_fit, dim = 'density')
	pbar.finish()
	return tides_fit.assign_coords(lon=ds.lon, lat=ds.lat)