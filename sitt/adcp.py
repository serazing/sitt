import dask.delayed as delayed
import xarray as xr
import numpy as np

def latlon2yx(lat, lon):
    """
    Convert latitude and longitude arrays to y and x arrays in km
    """
    R_earth = 6371 * 1000
    y = (1 / 180.) * np.pi * R_earth * lat
    x = xr.ufuncs.cos(np.pi / 180. * lat) * np.pi / 180. * R_earth * lon
    return y, x


@delayed(pure=True)
def lagged_difference(a, dim, lag, order=1):
	return (a.shift(**{dim: lag}) - a) ** order


@delayed(pure=True)
def norm(dx, dy):
	return xr.ufuncs.sqrt(dx ** 2 + dy ** 2) / 1.e3


def structure_function(ds, max_lag, dim, order=2):
	y, x = latlon2yx(ds['lat'], ds['lon'])
	distance_list, d2u_list, d2v_list = [], [], []
	for lag in range(max_lag):
		dx = lagged_difference(x, dim, lag)
		dy = lagged_difference(y, dim, lag)
		d2u = lagged_difference(ds['u'], dim, lag, order=2)
		d2v = lagged_difference(ds['v'], dim, lag, order=2)
		distance = norm(dx, dy)
		distance_list.append(distance)
		d2u_list.append(d2u)
		d2v_list.append(d2v)
	r = delayed(xr.concat)(distance_list, dim='lags').stack(r=(dim, 'lags'))
	strucfunc_u = delayed(xr.concat)(d2u_list, dim='lags').stack(r=(dim,
	                                                                'lags'))
	strucfunc_v = delayed(xr.concat)(d2v_list, dim='lags').stack(r=(dim,
	                                                                'lags'))
	struct_func = xr.Dataset({'d2u': strucfunc_u.compute(),
		                      'd2v': strucfunc_v.compute()})
	struct_func = struct_func.assign_coords(r=r.compute())
	return struct_func