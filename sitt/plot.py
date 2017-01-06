# Numpy
import numpy as np
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import cmocean
# Cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
# Ipywidgets
import ipywidgets as ipyw
from IPython.display import display

def spectrum_plot(x, y, ax=None, **kwargs):
	"""
	Define a nice spectrum with twin x-axis, one with frequencies, the
	other one with periods, on a predefined axis object
	** kwargs : optional keyword arguments
		See the plot method in matplotlib documentation
	"""
	if ax is None:
		ax = plt.gca()
	if not 'xlog' in kwargs:
		xlog = False
	else:
		xlog = kwargs['xlog']
		del kwargs['xlog']
	if not 'ylog' in kwargs:
		ylog = False
	else:
		ylog = kwargs['ylog']
		del kwargs['ylog']
	if not 'xlim' in kwargs:
		xlim = None
	else:
		xlim = kwargs['xlim']
		del kwargs['xlim']
	if not 'ylim' in kwargs:
		ylim = None
	else:
		ylim = kwargs['ylim']
		del kwargs['ylim']
	ax.plot(x, y, **kwargs)
	if xlog:
		ax.set_xscale('log', nonposx='clip')
		try:
			xmin = np.ceil(np.log10(np.abs(xlim[0]))) - 1
			xmax = np.ceil(np.log10(np.abs(xlim[1])))
		except:
			xmin = np.ceil(np.log10(np.abs(x[1,]))) - 1
			xmax = np.ceil(np.log10(np.abs(x[-1,])))
		ax.set_xlim((10 ** xmin, 10 ** xmax))
	else:
		try:
			ax.set_xlim(xlim)
		except:
			ax.set_xlim(np.min(x), np.max(x))
	try:
		ax.set_ylim(ylim)
	except:
		pass
	if ylog:
		ax.set_yscale('log', nonposx='clip')
	ax.twiny = ax.twiny()
	if xlog:
		ax.twiny.set_xscale('log', nonposx='clip')
		ax.twiny.set_xlim((10 ** xmin, 10 ** xmax))
		new_major_ticks = 10 ** np.arange(xmin + 1, xmax, 1)
		new_major_ticklabels = 1. / new_major_ticks
		new_major_ticklabels = \
			["%.3g" % i for i in new_major_ticklabels]
		ax.twiny.set_xticks(new_major_ticks)
		ax.twiny.set_xticklabels(new_major_ticklabels, rotation=60,
		                           fontsize=12)
		A = np.arange(2, 10, 2)[np.newaxis]
		B = 10 ** (np.arange(-xmax, -xmin, 1)[np.newaxis])
		C = np.dot(B.transpose(), A)
		new_minor_ticklabels = C.flatten()
		new_minor_ticks = 1. / new_minor_ticklabels
		new_minor_ticklabels = \
			["%.3g" % i for i in new_minor_ticklabels]
		ax.twiny.set_xticks(new_minor_ticks, minor=True)
		ax.twiny.set_xticklabels(new_minor_ticklabels, minor=True,
		                           rotation=60, fontsize=12)
	ax.grid(True, which='both')


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
	land = cfeature.GSHHSFeature(scale='intermediate', levels=[1], facecolor=cfeature.COLORS['land'])
	ax.add_feature(land)
	ax.set_title(title)
	gl = ax.gridlines(draw_labels=True)
	gl.xlabels_top = False
	gl.ylabels_right = False
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	plt.tight_layout()


def nemo_view(dataset, title='', cmap=cmocean.cm.curl, transform=ccrs.PlateCarree()):

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
	cmap = cmocean.cm.balance
	cmap.set_bad('k',1.)
	data.plot.pcolormesh(ax=ax, cmap=cmap, **kwargs)
	ax.set_title(title)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)