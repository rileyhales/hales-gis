=====================
FAQ and Helpful Hints
=====================

Methods for downloading and viewing common spatial data

Opening GFS Data
~~~~~~~~~~~~~~~~
GFS files are complex and irregularly formatted grib files. When opening a gfs grib file in xarray, as is done by
geomatics, you will generally need to provide some additional kwargs which get used to filter out duplicate names and
data that would otherwise happen. Provide these with the xr_kwargs parameter and they will get passed to xarray through
geomatics. You will generally need to at least provide {'cfVarName': 'name_of_your_variable'}
