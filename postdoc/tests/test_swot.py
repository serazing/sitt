import matplotlib.pyplot as plt
from postdoc import swot
import cartopy.crs as ccrs

def test_add_swot_swath():
	kml_file = "/data/SWOT/SWOT_CalVal_june2015_Swath_10_60.kml"
	plt.figure()
	ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
	swot.add_swot_swath(kml_file, nadir=False, swath=True, ax=ax)
	swot.add_swot_swath(kml_file, nadir=True, swath=False, ax=ax)
