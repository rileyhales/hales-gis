from setuptools import setup

version = '0.8'

with open('README.md', 'r') as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

setup(
    name='geomatics',
    packages=['geomatics'],
    version=version,
    description='Geospatial tools for creating timeseries of from geospatial raster data in pure python.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    project_urls=dict(Documentation='https://geomatics.readthedocs.io',
                      Source='https://github.com/rileyhales/geomatics'),
    license='BSD 3-Clause',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
    ],
    install_requires=install_requires
)

# todo write and run tests on everything
# todo add notes in docs about default variable names, backend kwargs for gfs gribs, etc
# todo change filetype parameter to use checking for isinstance(xarray or h5 datasets)
# todo pass the hdf5_group param to downstream functions rather than alter var, can ruin setting dim order, indices, etc
# todo make the prj georeferencing info work on hdf5
