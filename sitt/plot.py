# Numpy
import numpy as np
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
# Cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
# Ipywidgets
import ipywidgets as ipyw
from IPython.display import display

def add_map(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,
            central_longitude=0., scale='auto', ax=None):
    """
    Add the map to the existing plot using cartopy

    Parameters
    ----------
    lon_min : float, optional
        Western boundary, default is -180
    lon_max : float, optional
        Eastern boundary, default is 180
    lat_min : float, optional
        Southern boundary, default is -90
    lat_max : float, optional
        Northern boundary, default is 90
    central_longitude : float, optional
        Central longitude, default is 180
    scale : {‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, ‘full’}, optional
        The map scale, default is 'auto'
    ax : GeoAxes, optional
        A new GeoAxes will be created if None

    Returns
    -------
    ax : GeoAxes
    Return the current GeoAxes instance
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    extent = (lon_min, lon_max, lat_min, lat_max)
    if ax is None:
        ax = plt.subplot(1, 1, 1,
                         projection=ccrs.PlateCarree(
	                                       central_longitude=central_longitude))
    ax.set_extent(extent)
    land = cfeature.GSHHSFeature(scale=scale,
                                 levels=[1],
                                 facecolor=cfeature.COLORS['land'])
    ax.add_feature(land)
    gl = ax.gridlines(draw_labels=True, linestyle=':', color='black',
                      alpha=0.5)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def nice_map(dataarray, ax=None, title='', transform=ccrs.PlateCarree(),
             **kwargs):
	"""
	Make a nice map with gridlines and coordinates

	Parameters
	----------
	ax: matplotlib axes object, optional
		If None, uses the current axis
	title: str, optional
		Title of the plot
	"""
	if ax is None:
		ax = plt.subplot(1, 1, 1, projection=transform)
	dataarray.plot.pcolormesh(ax=ax, transform=transform, **kwargs)
	land = cfeature.GSHHSFeature(scale='intermediate', levels=[1],
	                             facecolor=cfeature.COLORS['land'])
	ax.add_feature(land)
	ax.set_title(title)
	gl = ax.gridlines(draw_labels=True)
	gl.xlabels_top = False
	gl.ylabels_right = False
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	plt.tight_layout()


def nemo_view(dataset, title='', cmap='spectral', transform=ccrs.PlateCarree()):

	def map(var, time=0, lev=0, vmin=-1., vmax=1.):
		subset = dataset.isel(time_counter=time, deptht=lev)[var]
		nice_map(subset, title=title, cmap=cmap,
		         transform=transform, vmin=vmin, vmax=vmax)

	time_widget = ipyw.widgets.IntSlider(description = "Time", disabled = False, value=0,
	                                     min=0, max=len(dataset.time_counter))

	var_widget = ipyw.widgets.Dropdown(options=[var for var in dataset.data_vars],
	                                   description='Variables:', disabled=False,
	                                   button_style='')

	# Text box to define the min value
	min_widget = ipyw.widgets.FloatText(value=-1., description='Min:',
	                                   disabled=False, color='black')
	# Text box to define the max value
	max_widget = ipyw.widgets.FloatText(value=1., description='Max:',
	                                   disabled=False, color='black')

	def update_min_max(*args):
		min_widget.value = dataset[var_widget.value].min()
		max_widget.value = dataset[var_widget.value].max()

	var_widget.observe(update_min_max, 'value')


	ipyw.interact(map, var=var_widget, time=time_widget, lev=(0, len(dataset.deptht), 1), vmin=min_widget,
	              vmax=max_widget)


def hovmoller(data, ax=None, title='', xlim=None, ylim=None, **kwargs):
	"""
	Parameters
	---------
	data: DataArray
		Must be 2 dimensional
	ax: matplotlib axes object, optional
		If None, uses the current axis
	title: str, optional
		Title of the plot
	**kwargs:
	    Keywords arguments used in `xarray.plot.pcolormesh`
	"""
	if ax is None:
		ax = plt.gca()
	cmap = 'seismic'
	cmap.set_bad('k',1.)
	data.plot.pcolormesh(ax=ax, cmap=cmap, **kwargs)
	ax.set_title(title)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)