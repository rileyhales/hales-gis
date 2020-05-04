import os
import tempfile

import geomatics

# todo finish some tests


if __name__ == '__main__':
    tmpdir = tempfile.gettempdir()

    # get some gfs data
    # gfs_files = geomatics.data.download_noaa_gfs(tmpdir, 1)
    # print(gfs_files)
    gfs_files = ['/var/folders/v3/3qy1n8y915nbd8y4v4lvjt9c0000gs/T/gfs_2020042212_2020042218.grb']

    # collect a shapefile
    geojson = geomatics.data.get_livingatlas_geojson('Northern Africa')
    shp_path = os.path.join(tmpdir, 'north_africa_shapefile.shp')
    geomatics.convert.geojson_to_shapefile(geojson, shp_path)

    kwarg = {'filter_by_keys': {'cfVarName': 'al'}}
    # print(geomatics.timedata.point_series(gfs_files, 'al', (15, 15), xr_kwargs=kwarg))
    # print(geomatics.timedata.box_series(gfs_files, 'al', (10, 10, 20, 20), xr_kwargs=kwarg))
    print(geomatics.timedata.shp_series(gfs_files, 'al', shp_path, tvar='valid_time', xr_kwargs=kwarg))
    print(geomatics.convert.make_affine_transform(gfs_files[0]))
    os.removedirs(tmpdir)
