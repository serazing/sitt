# Numpy
import numpy as np
# Matplotlib
import matplotlib.pyplot as plt
import cmocean
# Cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature


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


def nice_map(ax=None, title=''):
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
		ax = plt.gca()
	land = cfeature.GSHHSFeature(scale='intermediate', levels=[1], facecolor=cfeature.COLORS['land'])
	ax.add_feature(land)
	ax.set_title(title)
	gl = ax.gridlines(draw_labels=True)
	gl.xlabels_top = False
	gl.ylabels_right = False
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	plt.tight_layout()


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