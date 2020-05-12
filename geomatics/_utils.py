import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr

__all__ = ['open_by_engine', 'array_by_engine', 'get_slicing_info', 'slice_array_cell', 'slice_array_range']


def open_by_engine(path: str, engine: str = 'xarray', backend_kwargs: dict = None) -> np.array:
    if backend_kwargs is None:
        backend_kwargs = dict()
    if engine == 'xarray':
        return xr.open_dataset(path, backend_kwargs=backend_kwargs)
    elif engine == 'cfgrib':
        return xr.open_dataset(path, engine='cfgrib', backend_kwargs=backend_kwargs)
    elif engine == 'netcdf4':
        return nc.Dataset(path)
    elif engine == 'h5py':
        return h5py.File(path, 'r')
    # elif engine == 'pygrib':
    #     return pygrib.open(path)
    # elif engine in ('PIL', 'pillow'):
    #     return Image
    elif engine == 'rasterio':
        return xr.open_rasterio(path)
    else:
        raise ValueError(f'Unsupported engine: {engine}')


def array_by_engine(open_file, engine: str, var: str, h5_group: str = None):
    if engine in ('xarray', 'cfgrib', 'netcdf4', 'rasterio'):
        return open_file[var].data
    elif engine == 'h5py':
        # not using open_file[:] because [:] can't slice string data but ... catches it all
        if h5_group is not None:
            open_file = open_file[h5_group]
        return open_file[...]
    # todo other engines for getting array
    # elif engine == 'pygrib':
    #     return pygrib.open(path)
    # elif engine in ('PIL', 'pillow'):
    #     return Image
    else:
        raise ValueError(f'Unsupported engine: {engine}')


# todo finish this
def get_slicing_info(path: str,
                     engine: str,
                     var: str,
                     x_var: str,
                     y_var: str,
                     t_var: str,
                     coords: tuple,
                     xr_kwargs: dict = None,
                     h5_group: str = None) -> dict:
    """
    Determines the order of spatio-temporal dimensions (x, y, t axes) of the var specified and the indices for slicing
    the array at a point or bounding box based on the shape of the arrays for the dimension variables (x_var, y_var,
    t_var)

    Args:
        path: the path to a file
        xr_kwargs: A dictionary of kwargs that you might need when opening complex grib files with xarray
        var: The name of a variable as it is stored in the file e.g. often 'temp' or 'T' instead of Temperature
        x_var: Name of the x coordinate variable used to spatial reference the array. Default: 'lon'
        y_var: Name of the y coordinate variable used to spatial reference the array. Default: 'lat'
        t_var: Name of the time coordinate variable used for time referencing the data. Default: 'time'
        coord_pairs: a tuple for each coordinate pair in the format ((x1, y1), (x2, y2), ...)

    Returns:
        dict(indices=list, dim_order=str, file_type=str)
    """
    # open the file to be read
    tmp_file = open_by_engine(path, engine, xr_kwargs)

    # get a list of the x&y coordinates
    x_steps = array_by_engine(tmp_file, engine, x_var)
    y_steps = array_by_engine(tmp_file, engine, y_var)

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
        shape = list(array_by_engine(tmp_file, engine, var).shape)
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
    print(coords)
    for coord in coords:
        print(coord)
        indices.append(((np.abs(x_steps - float(coord[0]))).argmin(), (np.abs(y_steps - float(coord[1]))).argmin()), )

    return dict(indices=indices, dim_order=dims)


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
