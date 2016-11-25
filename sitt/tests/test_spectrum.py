import sitt.spectrum as spec
import xarray as xr
import numpy as np


# TODO: Check the coordinates

def test_fft_real_1d():
	""" Compare the result from the spectrum._fft function to numpy.fft.rfft
	"""
	a = [0, 1, 0, 0]
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['x'])
	assert  np.array_equal(np.asarray(spectrum_array), np.fft.rfft(a))


def test_fft_complex_1d():
	""" Compare the result from the spectrum.fft function to numpy.fft.fft
	"""
	a = np.exp(2j * np.pi * np.arange(8) / 8)
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['x'])
	assert  np.array_equal(np.asarray(spectrum_array), np.fft.fft(a))


def test_fft_real_2d():
	""" Compare the result from the spectrum.fft function to numpy.fft.rfftn
	"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['y', 'z'])
	assert np.array_equal(np.asarray(spectrum_array), np.fft.rfftn(a,
	                                                               axes=(2, 1)))


def test_fft_complex_2d():
	""" Compare the result from the spectrum.fft function to
	numpy.fft.fftn
	"""
	a = np.outer(np.outer([0, 1, 0, 0], [0, 1j, 1j]), [0, 1, 1, 1])
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['y', 'z'])
	assert np.array_equal(np.asarray(spectrum_array),
                     np.fft.fftn(a, axes=(-2, -1)))
#def test_psd_2d():
#	dummy_array = xr.DataArray(np.random.random((2000, 2000)), dims=['x', 'y'])
#	chunked_array = dummy_array.chunk(chunks={'x': 50, 'y': 50})
#	spectrum_array = spec.psd(chunked_array, dims=['x', 'y'])

#def test_psd_1d():
#	dummy_array = xr.DataArray(np.random.random((2000, 2000)), dims=['x', 'y'])
#	chunked_array = dummy_array.chunk(chunks={'y': 500})
#	spectrum_array = spec.psd(chunked_array, dims=['x'])