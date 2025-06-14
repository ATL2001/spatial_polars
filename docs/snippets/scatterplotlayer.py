import geodatasets
from lonboard import Map
from spatial_polars import read_spatial

home_sales_df = read_spatial(geodatasets.get_path("geoda.home_sales"))  # (1)!

scatterplotlayer = home_sales_df.spatial.to_scatterplotlayer(  # (2)!
    fill_cmap_col="floors",  # (3)!
    fill_cmap_type="categorical",  # (4)!
    fill_cmap={1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)},  # (5)!
    radius=250,  # (6)!
    stroked=False,  # (7)!
)

scatterplotlayer_map = Map(layers=[scatterplotlayer])  # (8)!
scatterplotlayer_map
