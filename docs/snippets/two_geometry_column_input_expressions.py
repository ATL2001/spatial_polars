# --8<-- [start:setup]
import polars as pl

from spatial_polars import scan_spatial

lake_df = (
    scan_spatial("https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip")
    .select("name", "geometry")
    .collect(engine="streaming")
)
print(f"There are {len(lake_df)} rows in lake_df")

boundary_df = (
    scan_spatial(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    .select("SOVEREIGNT", "geometry")
    .collect(engine="streaming")
)

lake_boundary_df = (
    lake_df.spatial.join(
        other=boundary_df,
        how="inner",
        predicate="intersects",
        on="geometry",
        suffix="_boundary",
    )
    .select(
        pl.col("name"),
        pl.col("SOVEREIGNT"),
        pl.col("geometry"),
        pl.col("geometry_boundary"),
    )
    .sort("name")
)
print(lake_boundary_df)
# --8<-- [end:setup]


# --8<-- [start:intersection]
lake_boundary_map = (
    lake_boundary_df.filter(
        pl.col("SOVEREIGNT").is_in(["United States of America", "Canada"])  # (1)!
    )
    .with_columns(
        pl.struct(
            pl.col("geometry"),
            pl.col("geometry_boundary"),  # (2)!
        ).spatial.intersection()  # (3)!
    )
    .drop("geometry_boundary")  # (4)!
    .spatial.viz("geometry", polygon_kwargs={"auto_highlight": True})  # (5)!
)
lake_boundary_map
# --8<-- [end:interintersectionsects]
