# Xarray
import xarray as xr
# Numpy
import numpy as np
# Dask
import dask.array as da
# Internals
import copy


def ps(array, dims, delta=1, chunks=None):
	"""
	Compute the power spectrum

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
	ps_attrs = dict()
	spectrum, ps_coords, ps_dims = _fft(array, dims, delta, chunks)
	#TODO: Make the correct normalization for the power spectrum and check with the Parseval theorem
	ps = spectrum * da.conj(spectrum)
	ps_attrs['description'] = 'Power spectrum performed along dimension(s) '
	return xr.DataArray(ps, coords=ps_coords, dims=ps_dims, attrs=ps_attrs, name='ps')

def psd(array, dims, delta=1, chunks=None):
	#TODO: Wraps the ps function and make the correct normalization
	pass

def fft(array, dims, delta=1, chunks=None):
	"""
	Perform a Fast Fourrier Transform on xarray objects.

	Parameters
	----------
	array : xarray.DataArray
		Array from which compute the spectrum
	dims : str or sequence
		Dimensions along which to compute the spectrum
		chunks : int, tuple or dict, optional
			Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or ``{
			'x': 5, 'y': 5}``

	Returns
	-------
	spectrum:
		The result of the Fourier transform represented with a DataArray
	spectrum_coords: dict
		New coordinates associated with the spectral array
	spectrum_dims: list
		New dimensions assocaited with the spectral array


	Notes
	-----
	If the input data is real, a real fft is performed over the first
	dimension, which is faster. Then the transform over the remaining
	dimensions are computed with the classic fft.
	"""
	pass


def _fft(array, dims, delta=1, chunks=None):
	"""This function is for private use only.
	"""
	spectrum = array.chunk(chunks=chunks).data
	spectrum_coords = dict()
	spectrum_dims = tuple()
	chunks = copy.copy(spectrum.chunks)
	shape = spectrum.shape
	first = True
	for dim in dims:
		if dim in array.dims:
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
			raise Warning, "Cannot find dimension " + dim + "in DataArray"
	for dim in array.dims:
		if dim not in array.dims:
			spectrum_dims += (dim,)
			spectrum_coords[dim] = np.asarray(array.coords[dim])
	return spectrum, spectrum_coords, spectrum_dims


def _tapper():
	pass

def _detrend():
	pass

