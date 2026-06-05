import polars as pl

from spatial_polars import read_spatial

lake_df = read_spatial("https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip")  # (1)!
boundary_df = read_spatial("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")# (2)!

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
