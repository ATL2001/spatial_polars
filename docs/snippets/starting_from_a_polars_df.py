# --8<-- [start:setup]
import polars as pl
import shapely
from spatial_polars import SpatialFrame
# --8<-- [end:setup]

# --8<-- [start:from_point_coords]
df = pl.DataFrame(  # (1)!
    {
        "Place": ["Gateway Arch", "Monks Mound"],
        "x": [-90.18497, -90.06211],
        "y": [38.62456, 38.66072],
        "z": [0, 0],
    }
)
print("Before SpatialFrame.from_point_coords:")
print(df)
s_df1 = SpatialFrame.from_point_coords(df, x_col="x", y_col="y", z_col="z")  # (2)!
print("After SpatialFrame.from_point_coords:")
print(s_df1)
# --8<-- [end:from_point_coords]

# --8<-- [start:from_wkb]
arch_wkb = shapely.Point(-90.18497, 38.62456).wkb
monks_mound_wkb = shapely.Point(-90.06211, 38.66072).wkb
df = pl.DataFrame(  # (1)!
    {
        "Place": ["Gateway Arch", "Monks Mound"],
        "wkb": [arch_wkb, monks_mound_wkb],
    }
)
print("Before SpatialFrame.from_WKB:")
print(df)
s_df2 = SpatialFrame.from_WKB(df, "wkb")  # (2)!
print("After SpatialFrame.from_WKB:")
print(s_df2)
# --8<-- [end:from_wkb]

# --8<-- [start:from_wkt]
arch_wkt = shapely.Point(-90.18497, 38.62456).wkt
monks_mound_wkt = shapely.Point(-90.06211, 38.66072).wkt
df = pl.DataFrame(  # (1)!
    {
        "Place": ["Gateway Arch", "Monks Mound"],
        "wkt": [arch_wkt, monks_mound_wkt],
    }
)
print("Before SpatialFrame.from_WKT:")
print(df)
s_df3 = SpatialFrame.from_WKT(df, "wkt")  # (2)!
print("After SpatialFrame.from_WKT:")
print(s_df3)
# --8<-- [end:from_wkt]
