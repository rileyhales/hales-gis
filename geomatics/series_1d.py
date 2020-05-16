import numpy as np
import pandas as pd

from ._utils import _open_by_engine, _array_by_engine, _pick_engine, _check_var_in_dataset

__all__ = ['point_series']


# TIMESERIES FUNCTIONS
def point_series(files: list,
                 var: str,
                 dimension: str,
                 identifier: int,
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
        dimension: The name of the variable containing the labeling/identifying information for the 1D data in the `var`
        identifier: The value of the `dimension` which you're interested in
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

    # get information to slice the array with
    dim_order, idx = _slicing_info_1d(files[0], var, dimension, identifier, t_var, engine, xr_kwargs, h5_group)

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
        vs = _slice_index(_array_by_engine(opened_file, var, h5_group), dim_order, idx)
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


# Auxiliary utilities
def _slicing_info_1d(path: str,
                     var: str,
                     dimension: str,
                     id: int,
                     t_var: str,
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
    dim_vals = _array_by_engine(tmp_file, dimension)

    # if its a netcdf or grib, the dimensions should be included by xarray
    if engine in ('xarray', 'cfgrib', 'netcdf4', 'rasterio'):
        dims = list(tmp_file[var].dims)
        for i, dim in enumerate(dims):
            dims[i] = str(dim).replace(dimension, 'd').replace(t_var, 't')
        dims = str.join('', dims)

    # guess the dimensions based on the shape of the variable array and length of the x/y steps
    elif engine == 'hdf5':
        if h5_group is not None:
            tmp_file = tmp_file[h5_group]
        shape = list(_array_by_engine(tmp_file, engine, var).shape)
        for i, length in enumerate(shape):
            if length == len(dim_vals):
                shape[i] = 'd'
            else:
                shape[i] = 't'
        dims = str.join('', shape)
    else:
        dims = False

    tmp_file.close()

    # gather all the indices
    d_min = dim_vals.min()
    d_max = dim_vals.max()
    if id < d_min or id > d_max:
        raise ValueError(f'specified id value ({id}) is outside the bounds of the data: [{d_min}, {d_max}]')
    # if the exact value is in the list, use that. otherwise find the closest to it
    if id in dim_vals:
        index = list(dim_vals).index(id)
    else:
        # todo maybe raise some kind of warning here?
        index = (np.abs(dim_vals - id)).argmin()

    return dims, index


def _slice_index(array, dim_order, index):
    if dim_order == 'td':
        return array[:, index]
    elif dim_order == 'dt':
        return array[index, :]
    elif dim_order == 'd':
        return array[index]
    else:
        raise ValueError('Unrecognized order of dimensions, unable to slice array.')
