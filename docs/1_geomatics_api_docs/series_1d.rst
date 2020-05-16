*********
series_1d
*********

These timeseries generating functions create timeseries from multidimensional data array files (netCDF, grib, HDF5) for
a single variable that contains 1-dimensional data (or 2 if the second dimension is time). The 1 dimension is usually an
identifier number, a length, or a size. The single variable varies in value with respect to that 1 dimension.

Some examples of data commonly stored in this format include:

- stream discharge: flow in the array depends on the stream's identifying number (stored in the 1-dimension)
- streamflow velocity: the velocity data array varies with respect to a single other condition such as the stream's identifier or time.

Timeseries Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: geomatics.series_1d
	:members: point_series
