import os
import tempfile

import geomatics


# todo finish some tests


def test_getdata(tmpdir):
    # collect gfs data
    gfs_files = geomatics.data.download_noaa_gfs(tmpdir, 2)
    # collect a shapefile
    geojson = geomatics.data.get_livingatlas_geojson('Northern Africa')
    shp_path = os.path.join(tmpdir, 'tmpshp.shp')
    geomatics.convert.geojson_to_shapefile(geojson, shp_path)
    return gfs_files, shp_path


if __name__ == '__main__':
    tmpdir = tempfile.gettempdir()
    gfs_files, shp_path = test_getdata(tmpdir)
    print(geomatics.convert.make_affine_transform('/Users/rileyhales/SpatialData/THREDDS/gldas/raw/GLDAS_NOAH025_M.A201001.021.nc4'))
    os.removedirs(tmpdir)
