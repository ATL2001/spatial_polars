import polars as pl

from spatial_polars import scan_spatial

lake_lf = (
    scan_spatial("https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip")
    .select("name", "geometry")
)  # (1)!

boundary_lf = (
    scan_spatial(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    )
    .select("SOVEREIGNT", "geometry")
)  # (2)!

lake_df, boundary_df = pl.collect_all([lake_lf, boundary_lf])
print(f"There are {len(lake_df)} rows in lake_df")
print(f"There are {len(boundary_df)} rows in boundary_df")

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
