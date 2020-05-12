import glob
import os
import tempfile

import geomatics


def test_get_gfs():
    tmpdir = tempfile.gettempdir()
    gfs_files = geomatics.data.download_noaa_gfs(tmpdir, 1)
    print(gfs_files)
    return


def test_geojson_shapefile():
    geojson = geomatics.data.get_livingatlas_geojson('Northern Africa')
    shp_path = os.path.join('/Users/riley/spatialdata/shapefiles', 'north_africa_shapefile.shp')
    geomatics.vector.geojson_to_shapefile(geojson, shp_path)
    return


def test_netcdf_tools():
    netcdf_files = sorted(glob.glob('/Users/riley/spatialdata/thredds/gldas/raw/*.nc4'))
    pt_coords = (-112, 45)
    bx_coords = ((-115, 40), (-110, 45))
    shp_path = '/Users/riley/spatialdata/shapefiles/north_africa_shapefile.shp'
    # ts_nc_pt = geomatics.timedata.point_series(netcdf_files, 'Tair_f_inst', pt_coords, x_var='lon', y_var='lat')
    # ts_nc_bx = geomatics.timedata.box_series(netcdf_files, 'Tair_f_inst', bx_coords, x_var='lon', y_var='lat')
    # ts_nc_shp = geomatics.timedata.shp_series(netcdf_files, 'Tair_f_inst', shp_path, x_var='lon', y_var='lat')
    geomatics.vector.netcdf_to_geotiff(netcdf_files, 'Tair_f_inst', save_dir='/Users/riley/spatialdata/tmpdir/',
                                       x_var='lon', y_var='lat')
    # print(ts_nc_pt)
    # print(ts_nc_bx)
    # print(ts_nc_shp)
    return


def test_grib_tools():
    grib_files = sorted(glob.glob('/Users/riley/spatialdata/thredds/gfs/*.grb'))
    # geomatics.data.inspect_grib(grib_files[0], 423)
    shp_path = '/Users/riley/spatialdata/shapefiles/north_africa_shapefile.shp'
    pt_coords = (250, 45)
    bx_coords = ((245, 40), (255, 45))
    xr_kwargs = {'filter_by_keys': {'cfVarName': 'al', 'typeOfLevel': 'surface'}}
    ts_gb_pt = geomatics.times.point_series(grib_files, 'al', pt_coords, x_var='longitude', y_var='latitude',
                                            xr_kwargs=xr_kwargs)
    ts_gb_bx = geomatics.times.box_series(grib_files, 'al', bx_coords, x_var='longitude', y_var='latitude',
                                          xr_kwargs=xr_kwargs)
    ts_gb_shp = geomatics.times.shp_series(grib_files, 'al', shp_path, x_var='longitude', y_var='latitude',
                                           xr_kwargs=xr_kwargs)
    print(ts_gb_pt)
    print(ts_gb_bx)
    print(ts_gb_shp)
    return


def test_hdf5_tools():
    hdf5_files = sorted(glob.glob('/Users/riley/spatialdata/smap/*.h5'))
    ts_h5_pt = geomatics.times.point_series(hdf5_files, 'Soil_Moisture_Retrieval_Data_1km/soil_moisture_1km',
                                            coords=(41, -112),
                                            x_var='Soil_Moisture_Retrieval_Data_1km/latitude_1km',
                                            y_var='Soil_Moisture_Retrieval_Data_1km/longitude_1km',
                                            t_var='a/rangeBeginningDateTime')
    ts_h5_bx = geomatics.times.box_series(hdf5_files, 'Soil_Moisture_Retrieval_Data_1km/soil_moisture_1km',
                                          coords=((-115, 40), (-110, 45)),
                                          x_var='Soil_Moisture_Retrieval_Data_1km/latitude_1km',
                                          y_var='Soil_Moisture_Retrieval_Data_1km/longitude_1km',
                                          t_var='a/rangeBeginningDateTime')
    ts_h5_shp = geomatics.times.shp_series(hdf5_files, 'Soil_Moisture_Retrieval_Data_1km/soil_moisture_1km',
                                           '/Users/riley/spatialdata/shapefiles/north_africa_shapefile.shp',
                                           x_var='Soil_Moisture_Retrieval_Data_1km/latitude_1km',
                                           y_var='Soil_Moisture_Retrieval_Data_1km/longitude_1km',
                                           t_var='a/rangeBeginningDateTime')
    print(ts_h5_pt)
    print(ts_h5_bx)
    print(ts_h5_shp)


if __name__ == '__main__':
    # print('netcdf tests')
    # test_netcdf_tools()
    # print('grib tests')
    # test_grib_tools()
    # print('hdf5 tests')
    # test_hdf5_tools()

    netcdf_files = sorted(glob.glob('/Users/riley/spatialdata/thredds/gldas/raw/*.nc4'))
    pt_coords = (-112, 45)
    bx_coords = ((-115, 40), (-110, 45))
    print(geomatics.times.point_series(netcdf_files, 'Tair_f_inst', pt_coords))
    print(geomatics.times.box_series(netcdf_files, 'Tair_f_inst', bx_coords))
