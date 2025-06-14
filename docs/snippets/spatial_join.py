import polars as pl

from spatial_polars import scan_spatial

lake_df = (
    scan_spatial("https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip")
    .select("name", "geometry")
    .collect(engine="streaming")
)  # (1)!
print(f"There are {len(lake_df)} rows in lake_df")

boundary_df = (
    scan_spatial(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    .select("SOVEREIGNT", "geometry")
    .collect(engine="streaming")
)  # (2)!

lake_boundary_df = (
    lake_df.spatial.join(  # (3)!
        other=boundary_df,  # (4)!
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
print(lake_boundary_df)
