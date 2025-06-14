# --8<-- [start:setup]
import os
import tempfile

import geodatasets

from spatial_polars import read_spatial

df = read_spatial(geodatasets.get_path("geoda.chicago_commpop"))
# --8<-- [end:setup]

# --8<-- [start:shapefile]
output_shp_path = os.path.join(tempfile.gettempdir(), "spatial_polars.shp")

df.spatial.write_spatial(output_shp_path)
print(f"Wrote dataframe to {output_shp_path}")
# --8<-- [end:shapefile]

# --8<-- [start:geopackage]
output_gpkg_path = os.path.join(tempfile.gettempdir(), "spatial_polars.gpkg")

df.spatial.write_spatial(output_gpkg_path, layer="chicago_commpop")
print(f"Wrote dataframe to {output_gpkg_path}")
# --8<-- [end:geopackage]

# --8<-- [start:append]
df.spatial.write_spatial(output_gpkg_path, layer="chicago_commpop", append=True)
# --8<-- [end:append]

# --8<-- [start:geoparquet]
output_gpq_path = os.path.join(tempfile.gettempdir(), "spatial_polars.parquet")

df.spatial.write_geoparquet(output_gpq_path)
print(f"Wrote dataframe to {output_gpq_path}")
# --8<-- [end:geoparquet]
