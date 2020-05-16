import numpy as np
import pandas as pd

from ._utils import _open_by_engine, _array_by_engine, _pick_engine, _check_var_in_dataset, _array_to_stat_list

__all__ = ['point_series']


# TIMESERIES FUNCTIONS
def point_series(files: list,
                 var: str,
                 coords: tuple,
                 dims: tuple = ('lon', 'lat', 'depth'),
                 t_dim: str = 'time',
                 fill_value: int = -9999,
                 engine: str = None,
                 h5_group: str = None,
                 xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values at the grid cell closest to a specified point.

    Args:
        files: A list of absolute paths to netcdf, grib, or hdf5 files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        coords: A tuple of the coordinates for the location of interest in the order ((x1, y1, z1), (x2, y2, z2))
        dims: A tuple of the names of the (x, y, z) variables in the data files, the same you specified coords for.
            Defaults to common x, y, z variables names: ('lon', 'lat', 'depth')
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_dim: Name of the time variable if it is used in the files. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        engine: the python package used to power the file reading. Defaults to best for the type of input data
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine is None:
        engine = _pick_engine(files[0])

    # get information to slice the array with
    dim_order, slices = _slicing_info(files[0], var, (coords,), dims, t_dim, engine, xr_kwargs, h5_group)

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        ts = _array_by_engine(opened_file, t_dim, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # extract the appropriate values from the variable
        vs = _array_by_engine(opened_file, var, h5_group)[slices]
        if vs.ndim == 0:
            if vs == fill_value:
                vs = np.nan
            values.append(vs)
        elif vs.ndim == 1:
            vs[vs == fill_value] = np.nan
            for v in vs:
                values.append(v)
        else:
            raise ValueError('There are too many dimensions')
        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


def box_series(files: list,
               var: str,
               coords: tuple,
               dims: tuple = ('lon', 'lat', 'depth'),
               t_dim: str = 'time',
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
        coords: A tuple of the coordinates for the location of interest in the order (x, y, z)
        dims: A tuple of the names of the (x, y, z) variables in the data files, the same you specified coords for.
            Defaults to common x, y, z variables names: ('lon', 'lat', 'depth')
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_dim: Name of the time variable if it is used in the files. Default: 'time'
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        stat_type: The method to turn the values within a bounding box into a single value: mean, min, max, median
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine is None:
        engine = _pick_engine(files[0])

    # get information to slice the array with
    dim_order, slices = _slicing_info(files[0], var, (coords,), dims, t_dim, engine, xr_kwargs, h5_group)

    # make the return item
    times = []
    values = []

    # iterate over each file extracting the value and time for each
    for file in files:
        # open the file
        opened_file = _open_by_engine(file, engine, xr_kwargs)
        # get the times
        ts = _array_by_engine(opened_file, t_dim, h5_group=h5_group)
        if ts.ndim == 0:
            times.append(ts)
        else:
            for t in ts:
                times.append(t)

        # slice the variable's array, returns array with shape corresponding to dimension order and size
        vs = _array_by_engine(opened_file, var, h5_group=h5_group)[slices]
        vs[vs == fill_value] = np.nan
        values += _array_to_stat_list(vs, stat_type)
        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


# Auxiliary utilities
def _slicing_info(path: str,
                  var: str,
                  coords: tuple,
                  dims: tuple,
                  t_dim: str,
                  engine: str = None,
                  xr_kwargs: dict = None,
                  h5_group: str = None, ) -> tuple:
    if engine is None:
        engine = _pick_engine(path)
    # open the file to be read
    tmp_file = _open_by_engine(path, engine, xr_kwargs)

    # validate choice in variables
    if not _check_var_in_dataset(tmp_file, var, h5_group):
        raise ValueError(f'the variable "{var}" was not found in the file {path}')

    # get a list of the x&y coordinates
    x_steps = _array_by_engine(tmp_file, dims[0])
    if x_steps.ndim == 2:
        x_steps = x_steps[0, :]
    y_steps = _array_by_engine(tmp_file, dims[1])
    if y_steps.ndim == 2:
        y_steps = y_steps[:, 0]
    z_steps = _array_by_engine(tmp_file, dims[2])

    assert x_steps.ndim == 1
    assert y_steps.ndim == 1

    # if its a netcdf or grib, the dimensions should be included by xarray
    if engine in ('xarray', 'cfgrib', 'netcdf4', 'rasterio'):
        dim_order = list(tmp_file[var].dims)
    elif engine == 'hdf5':
        if h5_group is not None:
            tmp_file = tmp_file[h5_group]
        dim_order = [i.label for i in tmp_file[var].dims]
    else:
        raise ValueError(f'Unable to determine dims for engine: {engine}')

    for i, d in enumerate(dim_order):
        dim_order[i] = str(d).replace(dims[0], 'x').replace(dims[1], 'y').replace(dims[2], 'z').replace(t_dim, 't')
    dim_order = str.join('', dim_order)

    tmp_file.close()

    if coords is None:
        return dim_order, coords

    # gather all the indices
    indices = []
    x_min = x_steps.min()
    x_max = x_steps.max()
    y_min = y_steps.min()
    y_max = y_steps.max()
    z_min = z_steps.min()
    z_max = z_steps.max()
    for coord in coords:
        # first verify that the location is in the bounds of the coordinate variables
        x, y, z = coord
        x = float(x)
        y = float(y)
        z = float(z)
        if x < x_min or x > x_max:
            raise ValueError(f'specified x value ({x}) is outside the min/max range: [{x_min}, {x_max}]')
        if y < y_min or y > y_max:
            raise ValueError(f'specified y value ({y}) is outside the min/max range: [{y_min}, {y_max}]')
        if z < z_min or z > z_max:
            raise ValueError(f'specified z value ({z}) is outside the min/max range: [{z_min}, {z_max}]')
        # then calculate the indicies and append to the list of indices
        indices.append((
            (np.abs(x_steps - x)).argmin(), (np.abs(y_steps - y)).argmin(), (np.abs(z_steps - z)).argmin(),))

    # make a tuple of indices and slicing objects which we use to get the values from the arrays later
    if len(indices) == 1:  # then only one coord is given and we want to slice that
        slices_dict = dict(x=indices[0][0], y=indices[0][1], z=indices[0][2], t=slice(None))
    else:  # there were 2 coordinates and we want to slice over their bounding range
        x1 = min(indices[0][0], indices[1][0])
        x2 = max(indices[0][0], indices[1][0])
        y1 = min(indices[0][1], indices[1][1])
        y2 = max(indices[0][1], indices[1][1])
        z1 = min(indices[0][2], indices[1][2])
        z2 = max(indices[0][2], indices[1][2])
        slices_dict = dict(x=slice(x1, x2), y=slice(y1, y2), z=slice(z1, z2), t=slice(None))

    return dim_order, tuple([slices_dict[d] for d in dim_order])
