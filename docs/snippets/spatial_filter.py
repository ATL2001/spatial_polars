import geodatasets
import polars as pl
import shapely
from lonboard import viz

from spatial_polars import read_spatial

nyc_earnings_df = read_spatial(geodatasets.get_path("geoda.nyc_earnings"))  # (1)!
print(f"There are {len(nyc_earnings_df):,} rows in the dataframe before filter.")

nyc_earnings_df = nyc_earnings_df.with_columns(
    pl.col("geometry").spatial.reproject(4326)  # (2)!
)

polygon = shapely.Polygon(  # (3)!
    (
        (-73.89257606917118, 40.78508934389371),
        (-73.87251149764286, 40.78511666797557),
        (-73.86103571752412, 40.80049834119043),
        (-73.89517435900939, 40.79976075792865),
        (-73.89257606917118, 40.78508934389371),
    )
)

filtered_nyc_earnings_df = nyc_earnings_df.filter(
    pl.col("geometry").spatial.intersects(polygon)  # (4)!
)
print(
    f"There are {len(filtered_nyc_earnings_df):,} rows in the dataframe after filter."
)

lonboard_map = viz(  # (5)!
    polygon,  # (6)!
    polygon_kwargs={
        "get_fill_color": (0, 0, 0, 0),  # (7)!
        "get_line_color": (255, 0, 0, 255),
        "get_line_width": 35,
    },
)

filtered_polygonlayer = filtered_nyc_earnings_df.spatial.to_polygonlayer(  # (8)!
    fill_color=(0, 0, 255)
)

lonboard_map.layers = list(lonboard_map.layers) + [filtered_polygonlayer]  # (9)!
lonboard_map
