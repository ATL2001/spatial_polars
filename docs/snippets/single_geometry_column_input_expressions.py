import geodatasets
import polars as pl

from spatial_polars import scan_spatial

df = (
    scan_spatial(
        geodatasets.get_path("geoda.guerry")  # (1)!
    )
    .select(
        pl.col("geometry").spatial.area().alias("polyogn_area"),  # (2)!
        pl.col("geometry").spatial.length().alias("polyogn_perimeter"),  # (3)!
        pl.col("geometry").spatial.bounds().alias("bounds"),  # (4)!
        pl.col("geometry").spatial.centroid().alias("centroid"),  # (5)!
        pl.col("geometry"),  # (6)!
    )
    .collect(engine="streaming")
)  # (7)!
print(df)
