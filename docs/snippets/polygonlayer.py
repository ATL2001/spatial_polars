import geodatasets
from lonboard import Map
from palettable.colorbrewer.diverging import RdYlGn_11
import polars as pl


from spatial_polars import scan_spatial

nyc_earnings_df = (
    scan_spatial(geodatasets.get_path("geoda.nyc_earnings"))  # (1)!
    .filter(pl.col("COUNTYFP10") == 61)  # (2)!
    .collect(engine="streaming")
)

polygon_layer = nyc_earnings_df.spatial.to_polygonlayer(  # (3)!
    fill_cmap_col="CE03_14",  # (4)!
    fill_cmap_type="continuous",  # (5)!
    fill_cmap=RdYlGn_11,  # (6)!
    fill_normalize_cmap_col=True,  # (7)!
    elevation="CE03_14",  # (8)!
    wireframe=True,  # (9)!
)

polygonlayer_map = Map(layers=[polygon_layer])  # (10)!
polygonlayer_map
