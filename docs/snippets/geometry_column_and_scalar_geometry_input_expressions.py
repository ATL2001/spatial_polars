import geodatasets
import polars as pl
import shapely
from lonboard import Map
from palettable.colorbrewer.diverging import RdYlGn_11

from spatial_polars import scan_spatial

arch_pt = shapely.Point(-90.18497, 38.62456)  # (1)!

df = (
    scan_spatial(geodatasets.get_path("geoda.health"))  # (2)!
    .select(
        "countyname",  # (3)!
        "geometry",  # (4)!
        pl.col("geometry").spatial.distance(arch_pt).alias("dist_to_arch"),  # (5)!
    )
    .collect(engine="streaming")  # (6)!
)

polygon_layer = df.spatial.to_polygonlayer(  # (7)!
    fill_cmap_col="dist_to_arch",
    fill_cmap_type="continuous",
    fill_cmap=RdYlGn_11,
    fill_normalize_cmap_col=True,
    line_width_min_pixels=0.5,
)

arch_map = Map(layers=[polygon_layer])  # (8)!
arch_map
