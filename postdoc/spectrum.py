# Xarray
import xarray as xr
# Numpy
import numpy as np
# Dask
import dask.array as da
# Internals
import copy


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
	psd_attrs = dict()
	spectrum, psd_coords, psd_dims = _fft(array, dims, delta, chunks)
	#TODO: Make the correct normalization for the psd
	psd = spectrum * da.conj(spectrum)
	psd_attrs['description'] = 'Power spectral density performed along dimension(s) '
	return xr.DataArray(psd, coords=psd_coords, dims=psd_dims, attrs=psd_attrs, name='psd')


def _fft(array, dims, delta=1, chunks=None):
	"""
	Perform a Fast Fourrier Transform on xarray objects. This function is for private use only.

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
	spectrum:
		The result of the Fourier transform represented with a DataArray
	spectrum_coords: dict
		New coordinates associated with the spectral array
	spectrum_dims: list
		New dimensions assocaited with the spectral array
	"""
	spectrum = array.chunk(chunks=chunks).data
	spectrum_coords = dict()
	spectrum_dims = tuple()
	chunks = copy.copy(spectrum.chunks)
	shape = spectrum.shape
	first = True
	for dim in array.dims:
		if dim in dims:
			spectrum_dims += ('f_' + dim,)
			axis = array.get_axis_num(dim)
			if first and not np.iscomplexobj(spectrum):
				# The first FFT is performed on real numbers: the use of rfft is faster
				spectrum_coords['f_' + dim] = np.fft.rfftfreq(len(array[dim]), delta)
				spectrum = da.fft.rfft(spectrum.rechunk({axis: shape[axis]}), axis=axis) \
					.rechunk({axis: chunks[axis][0]})
			else:
				# The successive FFTs are performed on complex numbers: need to use classic fft
				spectrum_coords['f_' + dim] = np.fft.fftshift(np.fft.fftfreq(len(array[dim]), delta))
				spectrum = da.fft.fft(spectrum.rechunk({axis: shape[axis]}), axis=axis) \
					.rechunk({axis: chunks[axis][0]})
			first = False
		else:
			spectrum_dims += (dim,)
			spectrum_coords[dim] = np.asarray(array.coords[dim])
	return spectrum, spectrum_coords, spectrum_dims


def _tapper():
	pass


def _detrend():
	pass
