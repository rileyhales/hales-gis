import h5py
import netCDF4 as nc
import pygrib
import xarray as xr

__all__ = ['netcdf', 'grib', 'hdf5', 'geotiff']


def netcdf(path: str) -> None:
    """
    Prints some useful summary information about a netcdf

    Args:
        path: The path to a netcdf file.
    """
    nc_obj = nc.Dataset(path, 'r', clobber=False, diskless=True, persist=False)

    print("This is your netCDF python object")
    print(nc_obj)
    print()

    print(f"There are {len(nc_obj.variables)} variables")  # The number of variables
    print(f"There are {len(nc_obj.dimensions)} dimensions")  # The number of dimensions
    print()

    print('These are the global attributes of the netcdf file')
    print(nc_obj.__dict__)  # access the global attributes of the netcdf file
    print()

    print("Detailed view of each variable")
    print()
    for variable in nc_obj.variables.keys():  # .keys() gets the name of each variable
        print('Variable Name:  ' + variable)  # The string name of the variable
        print('The view of this variable in the netCDF python object')
        print(nc_obj[variable])  # How to view the variable information (netcdf obj)
        print('The data array stored in this variable')
        print(nc_obj[variable][:])  # Access the numpy array inside the variable (array)
        print('The dimensions associated with this variable')
        print(nc_obj[variable].dimensions)  # Get the dimensions associated with a variable (tuple)
        print('The metadata associated with this variable')
        print(nc_obj[variable].__dict__)  # How to get the attributes of a variable (dictionary)
        print()

    for dimension in nc_obj.dimensions.keys():
        print(nc_obj.dimensions[dimension].size)  # print the size of a dimension

    nc_obj.close()  # close the file connection to the file
    return


def grib(path: str, band_number: int = False) -> None:
    """
    Prints a summary of the information available when you open a grib with pygrib

    Args:
        path: The path to any grib file
        band_number: Optional band number
    """
    grib = pygrib.open(path)
    grib.seek(0)
    print('This is a summary of all the information in your grib file')
    print(grib.read())

    if band_number:
        print()
        print('The keys for this variable are:')
        print(grib[band_number].keys())
        print()
        print('The data stored in this variable are:')
        print(grib[band_number].values)
    return


def hdf5(path: str) -> None:
    """
    Prints lots of messages showing information about variables, dimensions, and metadata

    Args:
        path: The path to any HDF5 file
    """
    ds = h5py.File(path)
    print('The following groups/variables are contained in this HDF5 file')
    ds.visit(print)
    return


def geotiff(path: str) -> None:
    """
    Prints the information available when you open a geotiff with xarray

    Args:
        path: The path to any geotiff file
    """
    print(xr.open_rasterio(path))
    return
