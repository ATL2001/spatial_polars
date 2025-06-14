import geodatasets
import polars as pl

from spatial_polars import scan_spatial

geom = (
    scan_spatial(
        geodatasets.get_path("geoda.nyc_earnings")  # (1)!
    )
    .select(
        pl.col("geometry").spatial.to_shapely_array()  # (2)!
    )
    .head(1)
    .collect()
    .item()
)  # (3)!

print(f"Without reprojecting centroid of first geometry: {geom.centroid.coords[0]}")


geom = (
    scan_spatial(
        geodatasets.get_path("geoda.nyc_earnings")  # (4)!
    )
    .select(
        pl.col("geometry").spatial.reproject(4326).spatial.to_shapely_array()  # (5)!
    )
    .head(1)
    .collect()
    .item()
)  # (6)!

print(f"With reprojecting centroid of first geometry: {geom.centroid.coords[0]}")
