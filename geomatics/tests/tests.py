import glob
import os
import tempfile

import geomatics

if __name__ == '__main__':
    tmpdir = tempfile.gettempdir()

    # get some gfs data
    gfs_files = geomatics.data.download_noaa_gfs(tmpdir, 1)
    kwarg = {'filter_by_keys': {'cfVarName': 'al'}}
    print(gfs_files)

    # collect a shapefile
    geojson = geomatics.data.get_livingatlas_geojson('Northern Africa')
    shp_path = os.path.join(tmpdir, 'north_africa_shapefile.shp')
    geomatics.convert.geojson_to_shapefile(geojson, shp_path)

    # test the timeseries functions
    # print(geomatics.timedata.point_series(gfs_files, 'al', (15, 15), xr_kwargs=kwarg))
    # print(geomatics.timedata.box_series(gfs_files, 'al', (10, 10, 20, 20), xr_kwargs=kwarg))
    print(geomatics.timedata.shp_series(gfs_files, 'al', shp_path, t_var='valid_time', xr_kwargs=kwarg))

    # test the conversions to geotiff
    print(geomatics.convert.make_affine_transform(gfs_files[0]))
    os.removedirs(tmpdir)

files = glob.glob('/Users/riley/spatialdata/thredds/gldas/raw/*.nc4')
geomatics.convert.netcdf_to_geotiff(files, 'Tair_f_inst', save_dir='/Users/riley/spatialdata/', xvar='lat', yvar='lon')
