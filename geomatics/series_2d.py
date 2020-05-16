import numpy as np
import pandas as pd
import rasterstats

from ._utils import _open_by_engine, _array_by_engine, _pick_engine, _check_var_in_dataset
from .data import gen_affine

__all__ = ['point_series', 'box_series', 'shp_series']


# TIMESERIES FUNCTIONS
def point_series(files: list,
                 var: str,
                 coords: tuple,
                 x_var: str = 'lon',
                 y_var: str = 'lat',
                 t_var: str = 'time',
                 fill_value: int = -9999,
                 engine: str = None,
                 h5_group: str = None,
                 xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple (x_value, y_value) where the xy values are the location you want to extract values for
            in units of the x and y coordinate variable
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine is None:
        engine = _pick_engine(files[0])
    if engine == 'rasterio':
        x_var = 'x'
        y_var = 'y'

    # get information to slice the array with
    slicing_info = _slicing_info_2d(files[0], var, x_var, y_var, t_var, (coords,), engine, xr_kwargs, h5_group)
    dim_order = slicing_info['dim_order']
    x_idx = slicing_info['indices'][0][0]
    y_idx = slicing_info['indices'][0][1]

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        ts = _array_by_engine(opened_file, t_var, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # extract the appropriate values from the variable
        vs = _slice_cell(_array_by_engine(opened_file, var, h5_group), dim_order, x_idx, y_idx)
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
               x_var: str = 'lon',
               y_var: str = 'lat',
               t_var: str = 'time',
               fill_value: int = -9999,
               stat_type: str = 'mean',
               engine: str = None,
               h5_group: str = None,
               xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple of the format ((min_x_value, min_y_value), (max_x_value, max_y_value)) where the xy values
            are in units of the x and y coordinate variable
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The method to turn the values within a bounding box into a single value: mean, min, max, median
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine == 'rasterio':
        x_var = 'x'
        y_var = 'y'

    # get information to slice the array with
    slicing_info = _slicing_info_2d(files[0], var, x_var, y_var, t_var, coords, engine, xr_kwargs, h5_group)
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
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)

        # get the times
        ts = _array_by_engine(opened_file, t_var, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _slice_range(
            _array_by_engine(opened_file, var, h5_group=h5_group), dim_order, xmin_idx, ymin_idx, xmax_idx, ymax_idx)
        vs[vs == fill_value] = np.nan
        # add the results to the lists of values and times
        if vs.ndim == 1 or vs.ndim == 2:
            if stat_type == 'mean':
                values.append(np.mean(vs))
            elif stat_type == 'median':
                values.append(np.median(vs))
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
                elif stat_type == 'median':
                    values.append(np.median(v))
                elif stat_type == 'max':
                    values.append(np.max(v))
                elif stat_type == 'min':
                    values.append(np.min(v))
                else:
                    raise ValueError(f'Unrecognized statistic, {stat_type}. Use stat_type= mean, min or max')
        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def shp_series(files: list,
               var: str,
               shp_path: str,
               x_var: str = 'lon',
               y_var: str = 'lat',
               t_var: str = 'time',
               fill_value: int = -9999,
               stat_type: str = 'mean',
               engine: str = None,
               h5_group: str = None,
               xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        shp_path: An absolute path to the .shp file in a shapefile. Must be same coord system as the raster data.
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The method to turn the values within the shapefile into a single value: mean, min, max, median
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame

    Examples:
        .. code-block:: python

            data = geomatics.timedata.shp_series([list, of, file, paths], 'AirTemp', '/path/to/shapefile.shp')
    """
    if engine == 'rasterio':
        x_var = 'x'
        y_var = 'y'

    # get information to slice the array with
    slicing_info = _slicing_info_2d(files[0], var, x_var, y_var, t_var, None, engine, xr_kwargs, h5_group)
    dim_order = slicing_info['dim_order']

    # generate an affine transform used in zonal statistics
    affine = gen_affine(files[0], x_var, y_var, engine=engine, xr_kwargs=xr_kwargs, h5_group=h5_group)

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        ts = _array_by_engine(opened_file, t_var, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group)
        vs[vs == fill_value] = np.nan
        # modify the array as necessary
        if vs.ndim == 2:
            # if the values are in a 2D array, cushion it with a 3rd dimension so you can iterate
            vs = np.reshape(vs, [1] + list(np.shape(vs)))
            dim_order = 't' + dim_order
        if 't' in dim_order:
            # roll axis brings the time dimension to the front so we can iterate over it -- may not work as expected
            vs = np.rollaxis(vs, dim_order.index('t'))

        # do zonal statistics on everything
        for values_2d in vs:
            # actually do the gis to get the value within the shapefile
            stats = rasterstats.zonal_stats(shp_path, values_2d, affine=affine, nodata=np.nan, stats=stat_type)
            # if your shapefile has many polygons, you get many values back. average them.
            tmp = [i[stat_type] for i in stats if i[stat_type] is not None]
            values.append(sum(tmp) / len(tmp))

        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


# Auxiliary utilities
def _slicing_info_2d(path: str,
                     var: str,
                     x_var: str,
                     y_var: str,
                     t_var: str,
                     coords: tuple or None,
                     engine: str = None,
                     xr_kwargs: dict = None,
                     h5_group: str = None, ) -> dict:
    if engine is None:
        engine = _pick_engine(path)
    # open the file to be read
    tmp_file = _open_by_engine(path, engine, xr_kwargs)

    # validate choice in variables
    if not _check_var_in_dataset(tmp_file, var, h5_group):
        raise ValueError(f'the variable "{var}" was not found in the file {path}')

    # get a list of the x&y coordinates
    x_steps = _array_by_engine(tmp_file, x_var)
    y_steps = _array_by_engine(tmp_file, y_var)

    # if the coordinate data was stored in 2d arrays instead of 1d lists of steps
    if x_steps.ndim == 2:
        # select the first row
        x_steps = x_steps[0, :]
    if y_steps.ndim == 2:
        # select the first column
        y_steps = y_steps[:, 0]

    assert x_steps.ndim == 1
    assert y_steps.ndim == 1

    # if its a netcdf or grib, the dimensions should be included by xarray
    if engine in ('xarray', 'cfgrib', 'netcdf4', 'rasterio'):
        dims = list(tmp_file[var].dims)
        for i, dim in enumerate(dims):
            dims[i] = str(dim).replace(x_var, 'x').replace(y_var, 'y').replace(t_var, 't')
        dims = str.join('', dims)

    # guess the dimensions based on the shape of the variable array and length of the x/y steps
    elif engine == 'hdf5':
        if h5_group is not None:
            tmp_file = tmp_file[h5_group]
        shape = list(_array_by_engine(tmp_file, engine, var).shape)
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

    if coords is None:
        return dict(dim_order=dims)

    # gather all the indices
    indices = []
    x_min = x_steps.min()
    x_max = x_steps.max()
    y_min = y_steps.min()
    y_max = y_steps.max()
    for coord in coords:
        # first verify that the location is in the bounds of the coordinate variables
        x, y = coord
        x = float(x)
        y = float(y)
        if x < x_min or x > x_max:
            raise ValueError(f'specified x value ({x}) is outside the bounds of the data: [{x_min}, {x_max}]')
        if y < y_min or y > y_max:
            raise ValueError(f'specified x value ({y}) is outside the bounds of the data: [{y_min}, {y_max}]')
        # then calculate the indicies and append to the list of indices
        indices.append(((np.abs(x_steps - x)).argmin(), (np.abs(y_steps - y)).argmin()), )

    return dict(indices=indices, dim_order=dims)


def _slice_cell(array, dim_order, x_idx, y_idx):
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


def _slice_range(array, dim_order, xmin_index, ymin_index, xmax_index, ymax_index):
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
