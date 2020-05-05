import os

import numpy as np
import rasterio
from rasterio.enums import Resampling

__all__ = ['upsample']


def upsample(files: list, scale: float) -> list:
    """
    Performs array math to artificially increase the resolution of a geotiff. No interpolation of values. A scale
    factor of X means that the length of a horizontal and vertical grid cell decreases by X. Be careful, increasing the
    resolution by X increases the file size by ~X^2

    Args:
        files: A list of absolute paths to the appropriate type of files (even if len==1)
        scale: A positive integer used as the multiplying factor to increase the resolution.

    Returns:
        1. A list of paths to the geotiff files created
        2. A rasterio affine transformation used on the geotransform
    """
    # Read raster dimensions
    raster_dim = rasterio.open(files[0])
    width = raster_dim.width
    height = raster_dim.height
    lon_min = raster_dim.bounds.left
    lon_max = raster_dim.bounds.right
    lat_min = raster_dim.bounds.bottom
    lat_max = raster_dim.bounds.top
    # Geotransform for each resampled raster (east, south, west, north, width, height)
    affine_resampled = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width * scale, height * scale)
    # keep track of the new files
    new_files = []

    # Resample each GeoTIFF
    for file in files:
        rio_obj = rasterio.open(file)
        data = rio_obj.read(
            out_shape=(int(rio_obj.height * scale), int(rio_obj.width * scale)),
            resampling=Resampling.nearest
        )
        # Convert new resampled array from 3D to 2D
        data = np.squeeze(data, axis=0)
        # Specify the filepath of the resampled raster
        new_filepath = os.path.splitext(file)[0] + '_upsampled.tiff'
        new_files.append(new_filepath)
        # Save the GeoTIFF
        with rasterio.open(
                new_filepath,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                nodata=np.nan,
                crs=rio_obj.crs,
                transform=affine_resampled,
        ) as dst:
            dst.write(data, 1)

    return new_files
