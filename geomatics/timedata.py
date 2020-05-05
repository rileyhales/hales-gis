import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
import rasterstats

from .data import _smart_open

__all__ = ['point_series', 'box_series', 'shp_series', 'generate_timejoining_ncml']


# TIMESERIES FUNCTIONS
def point_series(files: list,
                 variable: str,
                 coordinates: tuple,
                 file_type: str = None,
                 x_var: str = 'longitude',
                 y_var: str = 'latitude',
                 t_var: str = 'time',
                 xr_kwargs: dict = {},
                 fill_value: int = -9999) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        variable: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coordinates: A tuple of the format (x_value, y_value) where the xy values are in units of the x and y
            coordinate variable
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files
        fill_value: The value used for filling no_data spaces in the array. Default: -9999

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.point_series([list, of, file, paths], 'AirTemp', (10, 20))
    """
    # get a list of the x&y coordinates using the first file as a reference
    xr_obj = _smart_open(files[0], file_type, xr_kwargs)
    x_steps = xr_obj[x_var][:].data
    y_steps = xr_obj[y_var][:].data
    # determine the index in the netcdf's coordinates for the xy coordinate provided
    x_index = (np.abs(x_steps - round(float(coordinates[0]), 2))).argmin()
    y_index = (np.abs(y_steps - round(float(coordinates[1]), 2))).argmin()
    dim_order = _get_dimension_order(xr_obj[variable].dims, x_var, y_var, t_var)
    xr_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _smart_open(file, file_type, xr_kwargs)
        ts = xr_obj[t_var].data

        # extract the correct values from the array and
        vs = _slice_point(xr_obj[variable], dim_order, x_index, y_index)
        vs[vs == fill_value] = np.nan

        # add the results to the lists of values and times depending on the size of the array extracted
        if vs.ndim == 0:
            values.append(vs)
        else:
            for v in vs:
                values.append(v)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)
        xr_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def box_series(files: list,
               variable: str,
               coordinates: tuple,
               file_type: str = None,
               x_var: str = 'longitude',
               y_var: str = 'latitude',
               t_var: str = 'time',
               xr_kwargs: dict = {},
               fill_value: int = -9999,
               stat_type: str = 'mean') -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        variable: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coordinates: A tuple of the format (min_x_value, min_y_value, max_x_value, max_y_value) where the xy values are
            in units of the x and y coordinate variable
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The method to turn the values within a bounding box into a single value. Eg mean, min, max

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.box_series([list, of, file, paths], 'AirTemp', (10, 20, 15, 25))
    """
    # get a list of the x&y coordinates using the first file as a reference
    xr_obj = _smart_open(files[0], file_type, xr_kwargs)
    x_steps = xr_obj[x_var][:].data
    y_steps = xr_obj[y_var][:].data
    # get the indices of the bounding box corners
    xmin = (np.abs(x_steps - float(coordinates[0]))).argmin()
    xmax = (np.abs(x_steps - float(coordinates[2]))).argmin()
    xmin_index = min(xmin, xmax)
    xmax_index = max(xmin, xmax)
    ymin = (np.abs(y_steps - float(coordinates[1]))).argmin()
    ymax = (np.abs(y_steps - float(coordinates[3]))).argmin()
    ymin_index = min(ymin, ymax)
    ymax_index = max(ymin, ymax)
    # which order are the dimensions for this variable
    dim_order = _get_dimension_order(xr_obj[variable].dims, x_var, y_var, t_var)
    xr_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _smart_open(file, file_type, xr_kwargs)
        ts = xr_obj[t_var].data

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        values_array = _slice_box(xr_obj[variable], dim_order, xmin_index, ymin_index, xmax_index, ymax_index)
        values_array[values_array == fill_value] = np.nan
        if stat_type == 'mean':
            vs = np.mean(values_array)
        elif stat_type == 'max':
            vs = np.max(values_array)
        elif stat_type == 'min':
            vs = np.min(values_array)
        else:
            raise ValueError(f'Unrecognized statistic, {stat_type}. Use stat_type= mean, min or max')

        # add the results to the lists of values and times
        if vs.ndim == 0:
            values.append(vs)
        else:
            for v in vs:
                values.append(v)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)
        xr_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def shp_series(files: list,
               variable: str,
               shp_path: str,
               file_type: str = None,
               x_var: str = 'longitude',
               y_var: str = 'latitude',
               t_var: str = 'time',
               xr_kwargs: dict = {},
               fill_value: int = -9999,
               stat_type: str = 'mean') -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        variable: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        shp_path: An absolute path to the .shp file in a shapefile. Must be same coord system as the raster data.
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.shp_series([list, of, file, paths], 'AirTemp', '/path/to/shapefile.shp')
    """
    # get a list of the x&y coordinates using the first file as a reference
    xr_obj = _smart_open(files[0], file_type, xr_kwargs)
    nc_xs = xr_obj.variables[x_var][:]
    nc_ys = xr_obj.variables[y_var][:]
    affine = rasterio.transform.from_origin(nc_xs.min(), nc_ys.max(), nc_ys[1] - nc_ys[0], nc_xs[1] - nc_xs[0])
    dim_order = _get_dimension_order(xr_obj[variable].dims, x_var, y_var, t_var)
    xr_obj.close()

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _smart_open(file, file_type, xr_kwargs)
        ts = xr_obj[t_var].data

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        values_array = xr_obj[variable][:].data
        # drop fill and no data entries
        values_array[values_array == fill_value] = np.nan

        # modify the array as necessary
        if values_array.ndim == 2:
            # if the values are in a 2D array, cushion it with a 3rd dimension so you can iterate
            values_array = np.reshape(values_array, [1] + list(np.shape(values_array)))
            dim_order = 't' + dim_order
        if 't' in dim_order:
            # roll axis brings the time dimension to the front so we can iterate over it
            values_array = np.rollaxis(values_array, dim_order.index('t'))

        # do zonal statistics on everything
        for values_2d in values_array:
            # todo test this
            # print(type(values_2d))
            # print(values_2d.ndim)
            # print(affine)
            # actually do the gis to get the value within the shapefile
            stats = rasterstats.zonal_stats(shp_path, values_2d, affine=affine, nodata=np.nan, stats=stat_type)
            # if your shapefile has many polygons, you get many values back. weighted average of those values.
            tmp = [i[stat_type] for i in stats if i[stat_type] is not None]
            values.append(sum(tmp) / len(tmp))

        # add the timesteps to the list of times
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        xr_obj.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


# NCML
def generate_timejoining_ncml(files: list, save_dir: str, time_interval: int) -> None:
    """
    Generates a ncml file which aggregates a list of netcdf files across the "time" dimension and the "time" variable.
    In order for the times displayed in the aggregated NCML dataset to be accurate, they must have a regular time step
    between measurments.

    Args:
        files: A list of absolute paths to netcdf files (even if len==1)
        save_dir: the directory where you would like to save the ncml
        time_interval: the time spacing between datasets in the units of the netcdf file's time variable
          (must be constont for ncml aggregation to work properly)

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.generate_timejoining_ncml('/path/to/netcdf/', '/path/to/save', 4)
    """
    ds = nc.Dataset(files[0])
    units_str = str(ds['time'].__dict__['units'])
    ds.close()

    # create a new ncml file by filling in the template with the right dates and writing to a file
    with open(os.path.join(save_dir, 'time_joined_series.ncml'), 'w') as ncml:
        ncml.write(
            '<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">\n' +
            '  <variable name="time" type="int" shape="time">\n' +
            '    <attribute name="units" value="' + units_str + '"/>\n' +
            '    <attribute name="_CoordinateAxisType" value="Time" />\n' +
            '    <values start="0" increment="' + str(time_interval) + '" />\n' +
            '  </variable>\n' +
            '  <aggregation dimName="time" type="joinExisting" recheckEvery="5 minutes">\n'
        )
        for file in files:
            ncml.write('    <netcdf location="' + file + '"/>\n')
        ncml.write(
            '  </aggregation>\n' +
            '</netcdf>'
        )
    return


# MODULE LEVEL AUXILIARY TOOLS
def _slice_point(xarray_variable, dim_order, x_index, y_index):
    if dim_order == 'txy':
        return xarray_variable[:, x_index, y_index].data
    elif dim_order == 'tyx':
        return xarray_variable[:, y_index, x_index].data

    elif dim_order == 'xyt':
        return xarray_variable[x_index, y_index, :].data
    elif dim_order == 'yxt':
        return xarray_variable[y_index, x_index, :].data

    elif dim_order == 'xty':
        return xarray_variable[x_index, :, y_index].data
    elif dim_order == 'ytx':
        return xarray_variable[y_index, :, x_index].data

    elif dim_order == 'xy':
        return xarray_variable[x_index, y_index].data
    elif dim_order == 'yx':
        return xarray_variable[y_index, x_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def _slice_box(xarray_variable, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
    if dim_order == 'txy':
        return xarray_variable[:, xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'tyx':
        return xarray_variable[:, ymin_index:ymax_index, xmin_index:xmax_index].data

    elif dim_order == 'xyt':
        return xarray_variable[xmin_index:xmax_index, ymin_index:ymax_index, :].data
    elif dim_order == 'yxt':
        return xarray_variable[ymin_index:ymax_index, xmin_index:xmax_index, :].data

    elif dim_order == 'xty':
        return xarray_variable[xmin_index:xmax_index, :, ymin_index:ymax_index].data
    elif dim_order == 'ytx':
        return xarray_variable[ymin_index:ymax_index, :, xmin_index:xmax_index].data

    elif dim_order == 'xy':
        return xarray_variable[xmin_index:xmax_index, ymin_index:ymax_index].data
    elif dim_order == 'yx':
        return xarray_variable[ymin_index:ymax_index, xmin_index:xmax_index].data
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def _get_dimension_order(dimensions, x_var, y_var, t_var):
    # check if the variable has 2 dimensions -- should be xy coordinates (spatial references)
    if len(dimensions) == 2:
        if dimensions == (x_var, y_var):
            return 'xy'
        elif dimensions == (y_var, x_var):
            return 'yx'
        else:
            raise ValueError('Unexpected dimension name. Specify with the xvar, yvar, tvar keyword arguments')
    # check if the variables has 3 dimensions ie xy and time coordinates (spatio-temporal references)
    elif len(dimensions) == 3:
        # cases where time is first
        if dimensions == (t_var, x_var, y_var):
            return 'txy'
        elif dimensions == (t_var, y_var, x_var):
            return 'tyx'

        # cases where time is last
        elif dimensions == (x_var, y_var, t_var):
            return 'xyt'
        elif dimensions == (y_var, x_var, t_var):
            return 'yxt'

        # unlikely, but, if time is in the middle
        elif dimensions == (x_var, t_var, y_var):
            return 'xty'
        elif dimensions == (y_var, t_var, x_var):
            return 'ytx'

        else:
            raise ValueError('Unexpected dimension name(s). Specify with the xvar, yvar, tvar keyword arguments')
    else:
        raise ValueError('Your data should have either 2 (x,y) or 3 (x,y,time) dimensions')
