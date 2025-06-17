from .io import spatial_series_dtype, scan_spatial, read_spatial
from .spatialframe import SpatialFrame
from .spatialseries import SpatialSeries
from .spatialexpr import SpatialExpr

__all__ = [
    "spatial_series_dtype",
    "scan_spatial",
    "read_spatial",
    "SpatialFrame",
    "SpatialSeries",
    "SpatialExpr",
]

__version__ = "0.1.3" # dont forget pyproject.toml and uv lock