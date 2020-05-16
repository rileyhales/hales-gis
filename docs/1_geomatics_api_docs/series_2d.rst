*********
series_2d
*********

These timeseries generating functions create timeseries from multidimensional data array files (netCDF, grib, HDF5, GeoTiff) for
a single variable that contains 2-dimensional data (or 3 if the second dimension is time). The 2 dimensions are most commonly
geospatial referencing information: that is, the x and y coordinates of a location (e.g. latitude and longitude). The data in the
single variable changes with respect to the 2 dimensions and perhaps also time depending on the structure of the data.

Some examples of data commonly stored in this format include:

- geospatial raster/gridded data
- meteorological observation/simulation data

Timeseries Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: geomatics.series_2d
	:members: point_series, box_series,	shp_series
