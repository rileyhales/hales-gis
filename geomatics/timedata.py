import datetime
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
import rasterstats

from .data import detect_type, _smart_open, _get_array

__all__ = ['point_series', 'box_series', 'shp_series', 'generate_timejoining_ncml', 'get_slicing_info',
           'slice_array_cell', 'slice_array_range']


# TIMESERIES FUNCTIONS
def point_series(files: list,
                 var: str,
                 coords: tuple,
                 file_type: str = None,
                 x_var: str = 'longitude',
                 y_var: str = 'latitude',
                 t_var: str = 'time',
                 time_from_name: str = None,
                 h5_group: str = None,
                 xr_kwargs: dict = None,
                 fill_value: int = -9999) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple (x_value, y_value) where the xy values are the location you want to extract values for
            in units of the x and y coordinate variable
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        time_from_name: a string used by datetime.strptime to get a datetime object from the name of the file
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        fill_value: The value used for filling no_data spaces in the array. Default: -9999

    Returns:
        pandas.DataFrame
    """
    if h5_group:
        var = f'{h5_group}/{var}'
        x_var = f'{h5_group}/{x_var}'
        y_var = f'{h5_group}/{y_var}'
        t_var = f'{h5_group}/{t_var}'

    # get information to slice the array with
    slicing_info = get_slicing_info(files[0], file_type, xr_kwargs, var, x_var, y_var, t_var, (coords,))
    dim_order = slicing_info['dim_order']
    x_idx = slicing_info['indices'][0]
    y_idx = slicing_info['indices'][1]

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        if file_type is None:
            ft = detect_type(file)
        else:
            ft = file_type

        # open the file
        opened_file = _smart_open(file, ft, xr_kwargs)

        # handle the times for the file
        if time_from_name:
            ts.append(datetime.datetime.strptime(file, time_from_name))
        else:
            ts = _get_array(opened_file, ft, t_var)
            if ts.ndim == 0:
                times.append(ts)
            else:
                for t in ts:
                    times.append(t)

        # extract the appropriate values from the variable
        vs = slice_array_cell(_get_array(opened_file, ft, var), dim_order, x_idx, y_idx)
        if vs.ndim == 0:
            if vs == fill_value:
                vs = np.nan
            values.append(vs)
        else:
            vs[vs == fill_value] = np.nan
            for v in vs:
                values.append(v)

        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def box_series(files: list,
               var: str,
               coords: tuple,
               file_type: str = None,
               x_var: str = 'longitude',
               y_var: str = 'latitude',
               t_var: str = 'time',
               time_from_name: str = None,
               h5_group: str = None,
               xr_kwargs: dict = None,
               fill_value: int = -9999,
               stat_type: str = 'mean') -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple of the format ((min_x_value, min_y_value), (max_x_value, max_y_value)) where the xy values
            are in units of the x and y coordinate variable
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        time_from_name: a string used by datetime.strptime to get a datetime object from the name of the file
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The method to turn the values within a bounding box into a single value. Eg mean, min, max

    Returns:
        pandas.DataFrame
    """
    if h5_group:
        var = f'{h5_group}/{var}'
        x_var = f'{h5_group}/{x_var}'
        y_var = f'{h5_group}/{y_var}'
        t_var = f'{h5_group}/{t_var}'

    # get information to slice the array with
    slicing_info = get_slicing_info(files[0], file_type, xr_kwargs, var, x_var, y_var, t_var, (coords,))
    dim_order = slicing_info['dim_order']
    xmin_idx = min(slicing_info['indices'][0][0], slicing_info['indices'][1][0])
    xmax_idx = max(slicing_info['indices'][0][0], slicing_info['indices'][1][0])
    ymin_idx = min(slicing_info['indices'][0][1], slicing_info['indices'][1][1])
    ymax_idx = max(slicing_info['indices'][0][1], slicing_info['indices'][1][1])

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        if file_type is None:
            ft = detect_type(file)
        else:
            ft = file_type

        # open the file
        opened_file = _smart_open(file, ft, xr_kwargs)

        # handle the times for the file
        if time_from_name:
            ts.append(datetime.datetime.strptime(file, time_from_name))
        else:
            ts = _get_array(opened_file, ft, t_var)
            if ts.ndim == 0:
                times.append(ts)
            else:
                for t in ts:
                    times.append(t)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = slice_array_range(_get_array(opened_file, ft, var), dim_order, xmin_idx, ymin_idx, xmax_idx, ymax_idx)
        vs[vs == fill_value] = np.nan
        # add the results to the lists of values and times
        if vs.ndim == 1 or vs.ndim == 2:
            if stat_type == 'mean':
                values.append(np.mean(vs))
            elif stat_type == 'max':
                values.append(np.max(vs))
            elif stat_type == 'min':
                values.append(np.min(vs))
            else:
                raise ValueError(f'Unrecognized statistic, {stat_type}. Use stat_type= mean, min or max')
        else:
            for v in vs:
                if stat_type == 'mean':
                    values.append(np.mean(v))
                elif stat_type == 'max':
                    values.append(np.max(v))
                elif stat_type == 'min':
                    values.append(np.min(v))
                else:
                    raise ValueError(f'Unrecognized statistic, {stat_type}. Use stat_type= mean, min or max')
        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


# todo finish shape series
def shp_series(files: list,
               var: str,
               shp_path: str,
               file_type: str = None,
               x_var: str = 'longitude',
               y_var: str = 'latitude',
               t_var: str = 'time',
               time_from_name: str = None,
               h5_group: str = None,
               xr_kwargs: dict = None,
               fill_value: int = -9999,
               stat_type: str = 'mean') -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        shp_path: An absolute path to the .shp file in a shapefile. Must be same coord system as the raster data.
        file_type: The format of the data in the list of file paths provided by the files argument
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        time_from_name: a string used by datetime.strptime to get a datetime object from the name of the file
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The stats method to turn the values within the bounding box into a single value. Default: 'mean'

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.shp_series([list, of, file, paths], 'AirTemp', '/path/to/shapefile.shp')
    """
    # get information to slice the array with
    slicing_info = get_slicing_info(files[0], file_type, xr_kwargs, var, x_var, y_var, t_var, (coords,))
    dim_order = slicing_info['dim_order']
    # get a list of the x&y coordinates using the first file as a reference
    affine = rasterio.transform.from_origin(nc_xs.min(), nc_ys.max(), nc_ys[1] - nc_ys[0], nc_xs[1] - nc_xs[0])

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        xr_obj = _smart_open(file, file_type, xr_kwargs)
        ts = xr_obj[t_var].data

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        values_array = xr_obj[var][:].data
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


# FOR GETTING INFORMATION ABOUT THE ORGANIZATION OF THE FILE TO HELP WITH SLICING
def get_slicing_info(path, file_type, xr_kwargs, var, x_var, y_var, t_var, coord_pairs) -> dict:
    """
    Determines the order of spatio-temporal dimensions (x, y, t axes) of the var specified and the indices for slicing
    the array at a point or bounding box based on the shape of the arrays for the dimension variables (x_var, y_var,
    t_var)

    Args:
        path: the path to a file
        file_type: the file format of the provided path
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'longitude'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'latitude'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        coord_pairs: a tuple in the format ((x1, y1), (x2, y2), ...)

    Returns:
        dict(indices=list, dim_order=str, file_type=str)
    """
    # open the file to be read
    if file_type is None:
        file_type = detect_type(path)
    tmp_file = _smart_open(path, file_type, xr_kwargs)

    # get a list of the x&y coordinates
    x_steps = _get_array(tmp_file, file_type, x_var)
    y_steps = _get_array(tmp_file, file_type, y_var)

    # if the coordinate data was stored in 2d arrays instead of 1d lists of steps
    if x_steps.ndim == 2:
        # select the first row
        x_steps = x_steps[0, :]
    if y_steps.ndim == 2:
        # select the first column
        y_steps = y_steps[:, 0]

    assert x_steps.ndim == 1
    assert y_steps.ndim == 1

    # gather all the indices
    indices = []
    for coord in coord_pairs:
        indices.append((np.abs(x_steps - float(coord[0]))).argmin())
        indices.append((np.abs(y_steps - float(coord[1]))).argmin())

    # if its a netcdf or grib, the dimensions should be included
    if file_type == 'netcdf' or file_type == 'grib':
        dims = list(tmp_file[var].dims)
        for i, dim in enumerate(dims):
            dims[i] = str(dim).replace(x_var, 'x').replace(y_var, 'y').replace(t_var, 't')
        dims = str.join('', dims)

    # guess the dimensions based on the shape of the variable array and length of the x/y steps
    elif file_type == 'hdf5':
        shape = list(_get_array(tmp_file, file_type, var).shape)
        for i, length in enumerate(shape):
            if length == len(x_steps):
                shape[i] = 'x'
            elif length == len(y_steps):
                shape[i] = 'y'
            else:
                shape[i] = 't'
        dims = str.join('', shape)

    else:
        dims = False

    tmp_file.close()

    return dict(indices=indices, dim_order=dims, file_type=file_type)


def slice_array_cell(array, dim_order, x_idx, y_idx):
    if dim_order == 'txy':
        return array[:, x_idx, y_idx]
    elif dim_order == 'tyx':
        return array[:, y_idx, x_idx]

    elif dim_order == 'xyt':
        return array[x_idx, y_idx, :]
    elif dim_order == 'yxt':
        return array[y_idx, x_idx, :]

    elif dim_order == 'xty':
        return array[x_idx, :, y_idx]
    elif dim_order == 'ytx':
        return array[y_idx, :, x_idx]

    elif dim_order == 'xy':
        return array[x_idx, y_idx]
    elif dim_order == 'yx':
        return array[y_idx, x_idx]
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')


def slice_array_range(array, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
    if dim_order == 'txy':
        return array[:, xmin_index:xmax_index, ymin_index:ymax_index]
    elif dim_order == 'tyx':
        return array[:, ymin_index:ymax_index, xmin_index:xmax_index]

    elif dim_order == 'xyt':
        return array[xmin_index:xmax_index, ymin_index:ymax_index, :]
    elif dim_order == 'yxt':
        return array[ymin_index:ymax_index, xmin_index:xmax_index, :]

    elif dim_order == 'xty':
        return array[xmin_index:xmax_index, :, ymin_index:ymax_index]
    elif dim_order == 'ytx':
        return array[ymin_index:ymax_index, :, xmin_index:xmax_index]

    elif dim_order == 'xy':
        return array[xmin_index:xmax_index, ymin_index:ymax_index]
    elif dim_order == 'yx':
        return array[ymin_index:ymax_index, xmin_index:xmax_index]
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')
