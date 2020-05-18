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
    description='Geospatial tools for creating timeseries of from n-dimensional scientific data file formats',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    project_urls=dict(Documentation='https://geomatics.readthedocs.io',
                      Source='https://github.com/rileyhales/geomatics'),
    license='BSD 3-Clause',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
    ],
    install_requires=install_requires
)

# todo increment the version to 0.9 in setup and docs
# todo mod docs about kinds of stats you can get out including the 'all' option
# todo make the timeseries function generally applicable to the 1d and 2d data
# todo mod docs updated time from filename string
# todo guess the dimensions if they are not provided by the user?????
