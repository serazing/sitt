# Xarray
import xarray as xr
# Numpy
import numpy as np
# Dask
import dask.array as da
# Internals
import copy

# mlab
import matplotlib.mlab as mlab

#@xr.register_dataarray_accessor('spectrum')

def psd(array, dims, delta=1, chunks=None):
	"""
	Parameters
	----------
	array : xarray.DataArray
		Array from which compute the spectrum
	dims : str or sequence
		Dimensions along which to compute the spectrum
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or ``{'x': 5, 'y': 5}``
	Returns
	-------
	spectrum : xarray.DataArray
		Spectral array computed over the different arrays
	"""
	psd_coords = dict()
	psd_attrs = dict()
	psd_dims = tuple()
	dask_array = array.chunk(chunks=chunks).data
	first = True
	for dim in array.dims:
		if dim in dims:
			psd_dims += ('f_' + dim,)
			if first:
				# The first FFT is performed on real numbers: the use of rfft is faster
				psd_coords['f_' + dim] = np.fft.rfftfreq(len(array[dim]), delta)
				spectrum = da.fft.rfft(dask_array, axis=array.get_axis_num(dim))
				first = False
			else:
				# The successive FFTs are performed on complex numbers: need to use classic fft
				psd_coords['f_' + dim] = np.fft.fftshift(np.fft.fftfreq(len(array[dim]), delta))
				spectrum = da.fft.fft(spectrum, axis=array.get_axis_num(dim))
		else:
			psd_dims += (dim,)
			psd_coords[dim] = np.asarray(array.coords[dim])
	psd = spectrum * da.conj(spectrum)
	psd_attrs['description'] = 'Power spectral density performed along dimension(s) '
	return xr.DataArray(psd, coords=psd_coords, dims=psd_dims, attrs=psd_attrs, name='psd')

def _tapper():
	pass

def _detrend():
	pass

def _normalize():
	pass