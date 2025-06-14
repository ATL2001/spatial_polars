import geodatasets
from lonboard import Map

from spatial_polars import read_spatial

large_rivers_df = read_spatial(geodatasets.get_path("eea.large_rivers"))  # (1)!

pathlayer = large_rivers_df.spatial.to_pathlayer(  # (2)!
    auto_highlight=True,  # (3)!
    color=(0, 0, 255),  # (4)!
    width_min_pixels=5,  # (5)!
)

pathlayer_map = Map(layers=[pathlayer])  # (6)!
pathlayer_map
