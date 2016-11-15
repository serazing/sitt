# Import KML reader
from fastkml import kml
# Matplotlib
import matplotlib.pyplot as plt
# Cartopy
import cartopy.crs as ccrs
# Codecs
import codecs

def add_swot_swath(kml_file, pass_number=6, ax=None):
	"""Add the SWOT swath to the current or precised axes

	Parameters
	----------
	kml_file: string
		Full path of the KML file that contains the SWOT path
	pass_number: int, optional
		Number of the path to get, set to 6 by default (start at 0)
	ax: matplotlib axes, optional
		The axes object onto which add the SWOT path
	"""
	if ax == None:
		ax = plt.gca()
	f = codecs.open(filename, encoding='Latin-1')
	doc = f.read()
	doc_unicode = doc.encode("UTF-8")
	k = kml.KML()
	k.from_string(doc_unicode)
	features = list(k.features())
	f2 = list(features[0].features())
	# Get the right passage next to New Caledonia
	f2p = list(f2[pass_number].features())
	# Get nadir and swath KML objects
	fnadir = f2p[0]
	fswath = f2p[1]
	ax.add_geometries([fnadir.geometry], ccrs.PlateCarree(), edgecolor='red', color='none')
	ax.add_geometries([fswath.geometry], ccrs.PlateCarree(), facecolor='None', edgecolor='black')
