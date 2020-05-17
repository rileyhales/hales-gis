import numpy as np
import pandas as pd

from ._utils import _open_by_engine, _array_by_engine, _pick_engine, _check_var_in_dataset, _array_to_stat_list

__all__ = ['point', 'bounding_box', 'polygons', 'full_array_stats']


def point(files: list,
          var: str,
          coords: tuple,
          dims: tuple = None,
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
        coords: A tuple of the coordinates for the location of interest in the order (x, y, z)
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


def bounding_box(files: list,
                 var: str,
                 min_coords: tuple,
                 max_coords: tuple,
                 dims: tuple = None,
                 t_dim: str = 'time',
                 stat_type: str = 'mean',
                 fill_value: int = -9999,
                 engine: str = None,
                 h5_group: str = None,
                 xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        min_coords: A tuple of the minimum coordinates for the region of interest in the order (x, y, z)
        max_coords: A tuple of the maximum coordinates for the region of interest in the order (x, y, z)
        dims: A tuple of the names of the (x, y, z) variables in the data files, the same you specified coords for.
            Defaults to common x, y, z variables names: ('lon', 'lat', 'depth')
            X dimension names are usually 'lon', 'longitude', 'x', or similar
            Y dimension names are usually 'lat', 'latitude', 'y', or similar
            Z dimension names are usually 'depth', 'elevation', 'z', or similar
        t_dim: Name of the time variable if it is used in the files. Default: 'time'
        stat_type: The method to turn the values within a bounding box into a single value: mean, min, max, median
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
    dim_order, slices = _slicing_info(files[0], var, (min_coords, max_coords), dims, t_dim, engine, xr_kwargs, h5_group)

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


# todo geojson, shapefiles, shapely -> geopandas and masking
def polygons(files: list,
             var: str,
             poly: str or dict,
             dims: tuple = None,
             t_var: str = 'time',
             stat_type: str = 'mean',
             fill_value: int = -9999,
             engine: str = None,
             h5_group: str = None,
             xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values within the boundaries of your polygon shapefile of the same coordinate system.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        poly: # todo what kinds of polygon data can we accept
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


# todo needs to be tested, maybe fix bugs
def full_array_stats(files: list,
                     var: str,
                     t_dim: str = 'time',
                     stat_type: str = 'mean',
                     fill_value: int = -9999,
                     engine: str = None,
                     h5_group: str = None,
                     xr_kwargs: dict = None, ) -> pd.DataFrame:
    """
    Creates a timeseries of values based on values within a bounding box specified by your coordinates.

    Args:
        files: A list of absolute paths to netcdf or gribs files (even if len==1)
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        t_dim: Name of the time variable if it is used in the files. Default: 'time'
        stat_type: The method to turn the values within a bounding box into a single value: mean, min, max, median
        fill_value: The value used for filling no_data spaces in the array. Default: -9999
        engine: the python package used to power the file reading
        h5_group: if all variables in the hdf5 file are in the same group, you can specify the name of the group here
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray

    Returns:
        pandas.DataFrame
    """
    if engine is None:
        engine = _pick_engine(files[0])

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
        vs = _array_by_engine(opened_file, var, h5_group=h5_group)
        vs[vs == fill_value] = np.nan
        values += _array_to_stat_list(vs, stat_type)
        opened_file.close()

    # return the data stored in a dataframe
    return pd.DataFrame(np.transpose(np.asarray([times, values])), columns=['times', 'values'])


# Auxiliary utilities
# todo guess the dimensions if they are not provided by the user
# todo detect number of coordinates, extract steps and compute indices for varying range of coordinates.
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
