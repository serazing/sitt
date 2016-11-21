import postdoc.spectrum as spec
import xarray as xr
import numpy as np


def test_fft_real_1d():
	""" Compare the result from the spectrum._fft function to numpy.fft.rfft
	"""
	dummy_array = xr.DataArray([0, 1, 0, 0], dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['x'])
	assert  np.array_equal(np.asarray(spectrum_array), np.fft.rfft([0, 1, 0, 0]))


def test_fft_complex_1d():
	""" Compare the result from the spectrum._fft function to numpy.fft.rfft
	"""
	#TODO: Check the coordinates
	dummy_array = xr.DataArray(np.exp(2j * np.pi * np.arange(8) / 8), dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['x'])
	assert  np.array_equal(np.asarray(spectrum_array), np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8)))

def test_fft_real_2d():
	""" Compare the result from the spectrum._fft function to numpy.fft.rfft
	"""
	dummy_array = xr.DataArray([0, 1, 0, 0], dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = spec._fft(chunked_array, dims=['x'])
	assert np.array_equal(np.asarray(spectrum_array), np.fft.rfftn([0, 1, 0, 0]))

#def test_psd_2d():
#	dummy_array = xr.DataArray(np.random.random((2000, 2000)), dims=['x', 'y'])
#	chunked_array = dummy_array.chunk(chunks={'x': 50, 'y': 50})
#	spectrum_array = spec.psd(chunked_array, dims=['x', 'y'])

#def test_psd_1d():
#	dummy_array = xr.DataArray(np.random.random((2000, 2000)), dims=['x', 'y'])
#	chunked_array = dummy_array.chunk(chunks={'y': 500})
#	spectrum_array = spec.psd(chunked_array, dims=['x'])