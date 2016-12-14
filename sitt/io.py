import xarray as xr
import glob


class Nemo(object):

	def __init__(self, path, decode_times=True):
		"""
		Build and read a Nemo simulation under `xarray.DataSet` objects. The files
		 'coordinates.nc', 'mask.nc', 'mesh_hgr.nc' and 'mesh_zgr.nc' are required.
		"""
		try:
			self.coordinates = (xr.open_dataset(path + "/coordinates.nc", decode_times=False).
			                    squeeze().drop(('time', 'z', 'nav_lev')).
			                    set_coords(('nav_lon', 'nav_lat'))
			                    )
		except:
			raise RuntimeError("Impossible to find coordinates.nc")
		try:
			self.mask = (xr.open_dataset(path + "/mask.nc", decode_times=False).
			             squeeze().drop(('t', 'time_counter')).
			             set_coords(('nav_lon', 'nav_lat', 'nav_lev'))
			             )
		except:
			raise RuntimeError("Impossible to find mask.nc")
		try:
			self.mesh_hgr = (xr.open_dataset(path + "/mesh_hgr.nc", decode_times=False).
			                 squeeze().drop(('t', 'time_counter', 'z', 'nav_lev')).
			                 set_coords(('nav_lon', 'nav_lat'))
			                 )
		except:
			raise RuntimeError("Impossible to find mesh_hgr.nc")
		try:
			self.mesh_zgr = (xr.open_dataset(path + "/mesh_zgr.nc", decode_times=False).
			                 squeeze().drop(('t', 'time_counter')).
			                 set_coords(('nav_lon', 'nav_lat', 'nav_lev'))
			                 )
		except:
			raise RuntimeError("Impossible to find mesh_zgr.nc")
		if glob.glob(path + "/*/*gridT.nc"):
			self.gridT = (xr.open_mfdataset(path + "/*/*gridT.nc",
			                                decode_times=decode_times,
			                                drop_variables=('nav_lon', 'nav_lat')).
			              assign_coords(nav_lon=self.coordinates.nav_lon,
			                            nav_lat=self.coordinates.nav_lat).
			              assign(mask=self.mask.tmask).
			              set_coords('nav_lev')
			              )
		if glob.glob(path + "/*/*gridU.nc"):
			self.gridU = (xr.open_mfdataset(path + "/*/*gridU.nc",
			                          decode_times=decode_times,
			                          drop_variables=('nav_lon', 'nav_lat')).
			              assign_coords(nav_lon=self.coordinates.nav_lon,
			                            nav_lat=self.coordinates.nav_lat).
			              assign(mask=self.mask.umask).
			              set_coords('nav_lev')
			              )
		if glob.glob(path + "/*/*gridV.nc"):
			self.gridV = (xr.open_mfdataset(path + "/*/*gridV.nc",
			                          decode_times=decode_times,
			                          drop_variables=('nav_lon', 'nav_lat')).
			              assign_coords(nav_lon=self.coordinates.nav_lon,
			                            nav_lat=self.coordinates.nav_lat).
			              assign(mask=self.mask.vmask).
			              set_coords('nav_lev')
			              )
		if glob.glob(path + "/*/*gridW.nc"):
			self.gridW = (xr.open_mfdataset(path + "/*/*gridW.nc",
			                          decode_times=decode_times,
			                          drop_variables=('nav_lon', 'nav_lat')).
			              assign_coords(nav_lon=self.coordinates.nav_lon,
			                            nav_lat=self.coordinates.nav_lat).
			              assign(mask=self.mask.fmask).
			              set_coords('nav_lev')
			              )