import postdoc.spectrum as spec
import xarray as xr
import numpy as np

def test_psd_2d():
	dummy_array = xr.DataArray(np.random.random((2000, 2000)), dims=['x', 'y'])
	#chunked_array = dummy_array.chunk(chunks={'x': 50, 'y': 50})
	spectrum_array = spec.psd(dummy_array, dims=['x', 'y'])

def test_psd_1d():
	dummy_array = xr.DataArray(np.random.random((2000, 2000)), dims=['x', 'y'])
	chunked_array = dummy_array.chunk(chunks={'y': 500})
	spectrum_array = spec.psd(chunked_array, dims=['x'])