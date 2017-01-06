import matplotlib.pyplot as plt
from sitt import swot
import cartopy.crs as ccrs

def test_add_swot_swath():
	plt.figure()
	ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
	swot.add_swot_swath(nadir=False, swath=True)
	swot.add_swot_swath(nadir=True, swath=False)
