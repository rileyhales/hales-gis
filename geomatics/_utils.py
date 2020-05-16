import h5py
import numpy as np
import xarray as xr
from PIL import TiffImagePlugin, Image

__all__ = ['_open_by_engine', '_array_by_engine', '_pick_engine', '_check_var_in_dataset']


def _open_by_engine(path: str, engine: str = None, backend_kwargs: dict = None) -> np.array:
    if engine is None:
        engine = _pick_engine(path)
    if backend_kwargs is None:
        backend_kwargs = dict()
    if engine == 'xarray':
        return xr.open_dataset(path, backend_kwargs=backend_kwargs)
    elif engine == 'cfgrib':
        return xr.open_dataset(path, engine='cfgrib', backend_kwargs=backend_kwargs)
    elif engine == 'h5py':
        return h5py.File(path, 'r')
    elif engine in ('PIL', 'pillow'):
        return Image.open(path, 'r')
    elif engine == 'rasterio':
        return xr.open_rasterio(path)
    else:
        raise ValueError(f'Unable to open file, unsupported engine: {engine}')


def _array_by_engine(open_file, var: str, h5_group: str = None):
    if isinstance(open_file, xr.Dataset):  # xarray
        return open_file[var].data
    elif isinstance(open_file, h5py.Dataset):  # h5py
        # not using open_file[:] because [:] can't slice string data but ... catches it all
        if h5_group is not None:
            open_file = open_file[h5_group]
        return open_file[...]
    elif isinstance(open_file, TiffImagePlugin.TiffImageFile):  # geotiff
        return np.array(open_file)
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _pick_engine(path: str) -> str:
    if path.endswith('.nc') or path.endswith('.nc4'):
        return 'xarray'
    elif path.endswith('.grb') or path.endswith('.grib'):
        return 'cfgrib'
    elif path.endswith('.gtiff') or path.endswith('.tiff') or path.endswith('tif'):
        return 'rasterio'
    elif path.endswith('.h5') or path.endswith('.hd5') or path.endswith('.hdf5'):
        return 'h5py'
    else:
        raise ValueError(f'File does not match known files extension patterns: {path}')


def _check_var_in_dataset(open_file, variable, h5_group):
    # if isinstance(variable, str):
    #     variable = [variable, ]
    if isinstance(open_file, xr.Dataset):  # xarray
        return bool(variable in open_file.variables)
    elif isinstance(open_file, h5py.Dataset):  # h5py
        # not using open_file[:] because [:] can't slice string data but ... catches it all
        if h5_group is not None:
            open_file = open_file[h5_group]
        return bool(variable in open_file.keys())
    elif isinstance(open_file, TiffImagePlugin.TiffImageFile):  # geotiff
        return False
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')
