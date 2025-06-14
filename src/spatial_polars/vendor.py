from functools import lru_cache

import numpy as np
import shapely
from pyproj import Transformer

all = [
    "TransformerFromCRS",
    "transform",
]

TransformerFromCRS = lru_cache(Transformer.from_crs)


def transform(data, func):
    # https://github.com/geopandas/geopandas/blob/431b09a5a15a038504e2342f3a546161675e76ae/geopandas/array.py#L1756
    has_z = shapely.has_z(data)

    result = np.empty_like(data)

    coords = shapely.get_coordinates(data[~has_z], include_z=False)
    new_coords_z = func(coords[:, 0], coords[:, 1])
    result[~has_z] = shapely.set_coordinates(
        data[~has_z].copy(), np.array(new_coords_z).T
    )

    coords_z = shapely.get_coordinates(data[has_z], include_z=True)
    new_coords_z = func(coords_z[:, 0], coords_z[:, 1], coords_z[:, 2])
    result[has_z] = shapely.set_coordinates(
        data[has_z].copy(), np.array(new_coords_z).T
    )
    return result
