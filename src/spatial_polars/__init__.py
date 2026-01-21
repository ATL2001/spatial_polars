"""Spatial Polars."""

from .io import read_spatial, scan_spatial, spatial_series_dtype
from .spatialexpr import SpatialExpr
from .spatialframe import SpatialFrame
from .spatialseries import SpatialSeries

__all__ = [
    "SpatialExpr",
    "SpatialFrame",
    "SpatialSeries",
    "read_spatial",
    "scan_spatial",
    "spatial_series_dtype",
]

__version__ = "0.2.3"  # dont forget pyproject.toml and uv lock
