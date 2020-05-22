import glob

import geomatics

gldas_files = glob.glob('/Users/riley/spatialdata/thredds/gldas/raw/*.nc4')
# ts = geomatics.timeseries.point(gldas_files, 'Tair_f_inst', (-112, 42), ('lon', 'lat'))
# nwm_files = glob.glob('/Users/riley/Downloads/nwm*.nc')
# ts = geomatics.timeseries.point(nwm_files, 'streamflow', (1000, ), ('feature_id', ))
print(geomatics.convert.to_multiband_geotiff(gldas_files, 'Tair_f_inst', save_dir='/Users/riley/spatialdata/'))

import xarray

ds = xarray.open_rasterio('/Users/riley/spatialdata/multiband_collection.tif')
print(ds)
