import polars as pl
from spatial_polars import scan_spatial

lake_lf = scan_spatial("https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip") # (1)!
boundary_lf = scan_spatial("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")  # (2)!

lake_boundary_lf = (
    lake_lf.spatial.join(  # (3)!
        other=boundary_lf,  # (4)!
        how="inner",  # (5)!
        predicate="intersects",  # (6)!
        on="geometry",  # (7)!
        suffix="_boundary",  # (8)!
    )
    .select(
        pl.col("name"),  # (9)!
        pl.col("SOVEREIGNT"),
        pl.col("geometry"),
        pl.col("geometry_boundary"),
    )
    .sort("name")  # (10)!
)

lake_boundary_df = lake_boundary_lf.collect(engine="in-memory")  # (11)!
print(lake_boundary_df)
