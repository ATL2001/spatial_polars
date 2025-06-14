import geodatasets
from lonboard import Map
from palettable.colorbrewer.diverging import RdYlGn_11
import polars as pl

from spatial_polars import scan_spatial

nyc_earnings_grouped_layer = (
    scan_spatial(geodatasets.get_path("geoda.nyc_earnings"))  # (1)!
    .group_by(pl.col("COUNTYFP10"))  # (2)!
    .agg(
        pl.col("CE03_14").sum(),  # (3)!
        pl.len().alias("original_row_cnt"),  # (4)!
        pl.col("geometry").spatial.union_all(),  # (5)!
    )
    .collect(engine="streaming")  # (6)!
    .spatial.to_polygonlayer(  # (7)!
        auto_highlight=True,
        fill_cmap_col="CE03_14",
        fill_cmap_type="continuous",
        fill_cmap=RdYlGn_11,
    )
)

nyc_earnings_grouped_map = Map(layers=[nyc_earnings_grouped_layer])  # (8)!
nyc_earnings_grouped_map
