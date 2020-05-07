import os

import netCDF4 as nc
import numpy as np
import pygrib
import rasterio
import shapefile

from .prj import affine_from_netcdf, affine_from_grib

__all__ = ['geojson_to_shapefile', 'netcdf_to_geotiff', 'grib_to_geotiff']


def geojson_to_shapefile(geojson: dict, savepath: str) -> None:
    """
    Turns a valid dict, json, or geojson containing polygon data in a geographic coordinate system into a shapefile

    Args:
        geojson: a valid geojson as a dictionary or json python object. try json.loads for strings
        savepath: the full file path to save the shapefile to, including the file_name.shp
    """
    # create the shapefile
    fileobject = shapefile.Writer(target=savepath, shpType=shapefile.POLYGON, autoBalance=True)

    # label all the columns in the .dbf
    geomtype = geojson['features'][0]['geometry']['type']
    if geojson['features'][0]['properties']:
        for attribute in geojson['features'][0]['properties']:
            fileobject.field(str(attribute), 'C', '30')
    else:
        fileobject.field('Name', 'C', '50')

    # add the geometry and attribute data
    for feature in geojson['features']:
        # record the geometry
        if geomtype == 'Polygon':
            fileobject.poly(polys=feature['geometry']['coordinates'])
        elif geomtype == 'MultiPolygon':
            for i in feature['geometry']['coordinates']:
                fileobject.poly(polys=i)

        # record the attributes in the .dbf
        if feature['properties']:
            fileobject.record(**feature['properties'])
        else:
            fileobject.record('unknown')

    # close writing to the shapefile
    fileobject.close()

    # create a prj file
    if savepath.endswith('.shp'):
        savepath.replace('.shp', '')
    with open(savepath + '.prj', 'w') as prj:
        prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                  'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

    return


def netcdf_to_geotiff(files: list,
                      variable: str,
                      crs: str = 'EPSG:4326',
                      x_var: str = 'longitude',
                      y_var: str = 'latitude',
                      fill_value: int = -9999,
                      save_dir: str = False,
                      delete_sources: bool = False) -> list:
    """
    Converts the array of data for a certain variable in a netcdf file to a geotiff.

    Args:
        files: A list of absolute paths to the appropriate type of files (even if len==1)
        variable: The name of a variable as it is stored in the netcdf e.g. 'temp' instead of Temperature
        crs: Coordinate Reference System used by rasterio.open(). An EPSG ID string such as 'EPSG:4326' or
            '+proj=latlong'
        x_var: Name of the x coordinate variable used to spatial reference the netcdf array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the netcdf array. Default: 'latitude'
        save_dir: The directory to store the geotiffs to. Default: directory containing the netcdfs.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        delete_sources: Allows you to delete the source netcdfs as they are converted. Default: False

    Returns:
        A list of paths to the geotiff files created
    """
    # determine the affine transformation
    affine = affine_from_netcdf(files[0], variable, x_var, y_var)

    # A list of all the files that get written which can be returned
    output_files = []

    # Create a geotiff for each netcdf in the list of files
    for file in files:
        # set the files to open/save
        save_path = os.path.join(save_dir, os.path.basename(file) + '.tif')
        output_files.append(save_path)

        # open the netcdf and get the data array
        nc_obj = nc.Dataset(file, 'r')
        array = np.asarray(nc_obj[variable][:])
        array = array[0]
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it
        array = np.flip(array, axis=0)
        nc_obj.close()

        # if you want to delete the source netcdfs as you go
        if delete_sources:
            os.remove(file)

        # write it to a geotiff
        with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=array.shape[0],
                width=array.shape[1],
                count=1,
                dtype=array.dtype,
                nodata=np.nan,
                crs=crs,
                transform=affine,
        ) as dst:
            dst.write(array, 1)

    return output_files


def grib_to_geotiff(files: list,
                    band_number: int,
                    crs: str = 'EPSG:4326',
                    fill_value: int = -9999,
                    save_dir: str = False,
                    delete_sources: bool = False) -> list:
    """
    Converts a certain band number in grib files to geotiffs. Assumes WGS1984 GCS.

    Args:
        files: A list of absolute paths to the appropriate type of files (even if len==1)
        band_number: the band number that the array of interest is located on
        save_dir: The directory to store the geotiffs to. Default: directory containing the gribs.
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        delete_sources: Allows you to delete the source gribs as they are converted. Default: False
        crs: Coordinate Reference System used by rasterio.open(). An EPSG ID string such as 'EPSG:4326' or
            '+proj=latlong'

    Returns:
        A list of paths to the geotiff files created
    """
    # determine the affine transformation
    affine = affine_from_grib(files[0])

    # A list of all the files that get written which can be returned
    output_files = []

    for file in files:
        save_path = os.path.join(save_dir, os.path.basename(file) + '.tif')
        output_files.append(save_path)
        grib = pygrib.open(file)
        grib.seek(0)
        array = grib[band_number].values
        array[array == fill_value] = np.nan  # If you have fill values, change the comparator to git rid of it

        with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=array.shape[0],
                width=array.shape[1],
                count=1,
                dtype=array.dtype,
                nodata=np.nan,
                crs=crs,
                transform=affine,
        ) as dst:
            dst.write(array, 1)

        # if you want to delete the source gribs as you go
        if delete_sources:
            os.remove(file)

    return output_files
