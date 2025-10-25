##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9


import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import mapping


class Rasterization:
    """This class is used to convert map to rasterized matrix."""

    @staticmethod
    def rasterize_polygons(polygons, boundary, cell_size) -> np.ndarray:
        """Convert `shapely.geometry.Polygon` to rasterized matrix."""
        x_min, x_max, y_min, y_max = boundary
        width = int((x_max - x_min) / cell_size)
        height = int((y_max - y_min) / cell_size)
        transform = from_origin(x_min, y_max, cell_size, cell_size)

        geoms = [mapping(polygon) for polygon in polygons]

        raster = rasterize(
            geoms, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8
        )

        return raster
