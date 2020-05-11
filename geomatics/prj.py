import rasterio

from .data import _smart_open

__all__ = ['georeferenced_grid_info', 'affine_from_netcdf', 'affine_from_grib', 'affine_from_hdf5']


def georeferenced_grid_info(file: str,
                            file_type: str = None,
                            x_var: str = 'longitude',
                            y_var: str = 'latitude',
                            xr_kwargs: dict = None) -> dict:
    """
    Determines the information needed to create an affine transformation for a geo-referenced data array.

    Args:
        file: the absolute path to a netcdf or grib file
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon' (longitude)
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat' (latitude)
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        A dictionary containing the information needed to create the affine transformation of a dataset.
    """
    # open the file to be read
    ds = _smart_open(file, file_type=file_type, backend_kwargs=xr_kwargs)
    x_data = ds[x_var].values
    y_data = ds[y_var].values

    return {
        'x_first_val': x_data[0],
        'x_last_val': x_data[-1],
        'x_min': x_data.min(),
        'x_max': x_data.max(),
        'x_num_values': x_data.size,
        'x_resolution': x_data[1] - x_data[0],

        'y_first_val': y_data[0],
        'y_last_val': y_data[-1],
        'y_min': y_data.min(),
        'y_max': y_data.max(),
        'y_num_values': y_data.size,
        'y_resolution': y_data[1] - y_data[0]
    }


def affine_from_netcdf(file: str,
                       var: str,
                       x_var: str = 'longitude',
                       y_var: str = 'latitude',
                       xr_kwargs: dict = None) -> rasterio.transform.from_bounds:
    """
    Creates an affine transform from the dimensions of the coordinate variable data in a netCDF file

    Args:
        file: An absolute paths to the data file
        var: The name of a variable as it is stored in the data file e.g. 'temp' instead of Temperature
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon' (longitude)
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat' (latitude)
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        rasterio.transform.from_bounds
    """
    raster = _smart_open(file, file_type='netcdf', backend_kwargs=xr_kwargs)
    lon = raster.variables[x_var][:]
    lat = raster.variables[y_var][:]
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()
    data = raster[var].values
    height = data.shape[0]
    width = data.shape[1]
    raster.close()
    return rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)


def affine_from_grib(file: str,
                     var: str,
                     x_var: str = 'longitude',
                     y_var: str = 'latitude',
                     xr_kwargs: dict = None) -> rasterio.transform.from_bounds:
    """
    Creates an affine transform from the dimensions of the coordinate variable data in a Grib file

    Args:
        file: An absolute paths to the data file
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        rasterio.transform.from_bounds
    """
    raster = _smart_open(file, file_type='grib', backend_kwargs=xr_kwargs)
    lon = raster.variables[x_var][:]
    lat = raster.variables[y_var][:]
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()
    data = raster[var].values
    height = data.shape[0]
    width = data.shape[1]
    raster.close()
    return rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)


def affine_from_hdf5(file: str,
                     x_var: str = 'longitude',
                     y_var: str = 'latitude',
                     h5_group: str = None) -> rasterio.transform.from_bounds:
    """
    Creates an affine transform from the dimensions of the coordinate variable data in an HDF5 file

    Args:
        file: A path to an HDF5 file
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here

    Returns:
        rasterio.transform.from_bounds
    """
    raster = _smart_open(file, file_type='hdf5')
    if h5_group:
        raster = raster[h5_group]

    lon = raster[x_var][:]
    lat = raster[y_var][:]
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()
    raster.close()
    return rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, len(lon[0, :]), len(lat[0, :]))
