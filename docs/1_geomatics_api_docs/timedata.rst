********
timedata
********

Timeseries Functions
~~~~~~~~~~~~~~~~~~~~
These functions will generate a timeseries from 1 netcdf file with many timesteps or many netcdf files each containing
a single timestep.

.. automodule:: geomatics.timedata
	:members: point_series, box_series,	shp_series

NCML
~~~~
A tool for creating a netCDF Markup Language (NCML) file based on the absolute file paths to datasets.

.. automodule:: geomatics.timedata
	:members: generate_timejoining_ncml
	:noindex:

Array Slicing Help
~~~~~~~~~~~~~~~~~~
Functions that help determine how to slice arrays with arbitrary sizes and numbers of dimensions

.. automodule:: geomatics.timedata
	:members: get_slicing_info
	:noindex:
