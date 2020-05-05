import rasterio

from .data import _smart_open

__all__ = ['georeferenced_grid_info', 'affine_trans_from_netcdf_file', 'affine_trans_from_grib_file']


def georeferenced_grid_info(file: str,
                            file_type: str = None,
                            x_var: str = 'longitude',
                            y_var: str = 'latitude',
                            xr_kwargs: dict = {}) -> dict:
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
    ds = _smart_open(file, filetype=file_type, backend_kwargs=xr_kwargs)
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


def affine_trans_from_netcdf_file(file: str, variable: str, x_var: str = 'longitude', y_var: str = 'latitude') -> dict:
    """

    Args:
        file: An absolute paths to the data file
        variable: The name of a variable as it is stored in the data file e.g. 'temp' instead of Temperature
        x_var:
        y_var:

    Returns:

    """
    # open the file
    raster = _smart_open(file, filetype='netcdf')

    lat = raster.variables[x_var][:]
    lon = raster.variables[y_var][:]
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()
    data = raster[variable].values
    height = data.shape[0]
    width = data.shape[1]

    raster.close()

    return rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)


def affine_trans_from_grib_file(file: str) -> dict:
    """

    Args:
        file: An absolute paths to the data file

    Returns:

    """
    raster = rasterio.open(file)
    width = raster.width
    height = raster.height
    lon_min = raster.bounds.left
    lon_max = raster.bounds.right
    lat_min = raster.bounds.bottom
    lat_max = raster.bounds.top
    return rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
