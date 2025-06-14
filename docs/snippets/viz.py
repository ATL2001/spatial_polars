import geodatasets

from spatial_polars import read_spatial

geoda_milwaukee1_df = read_spatial(geodatasets.get_path("geoda.milwaukee1"))  # (1)!
milwaukee_map = geoda_milwaukee1_df.spatial.viz()  # (2)!

milwaukee_map
