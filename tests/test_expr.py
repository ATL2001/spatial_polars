import numpy as np
import polars as pl
import shapely
from fixtures import *  # NOQA F403
from polars.testing import assert_series_equal


def test_reproject(arch_mound_df: pl.DataFrame) -> None:
    reprojected_shapely_array = (
        arch_mound_df.select(pl.col("geometry").spatial.reproject(32615))
        .to_series()
        .spatial.to_shapely_array()
    )

    expected_array = shapely.points(
        [(745063.298, 4278874.516), (755632.326, 4283223.681)]
    )

    geoms_match = shapely.equals_exact(reprojected_shapely_array, expected_array, 0.001)

    assert np.all(geoms_match)


def test_distance_scalar(two_points_df: pl.DataFrame) -> None:
    distance_s = two_points_df.select(
        pl.col("geometry").spatial.distance(shapely.Point(0, 1)).alias("distance")
    ).to_series()

    expected_s = pl.Series("distance", [1, 1], dtype=pl.Float64)

    assert_series_equal(distance_s, expected_s)


def test_distance_two_cols(
    two_points_df: pl.DataFrame, two_more_points_df: pl.DataFrame
) -> None:
    df = pl.concat(
        [two_points_df, two_more_points_df.rename({"geometry": "other_geometry"})],
        how="horizontal",
    )

    distance_s = df.select(
        pl.struct(pl.col("geometry"), pl.col("other_geometry"))
        .spatial.distance()
        .alias("distance")
    ).to_series()

    expected_s = pl.Series("distance", [10, 20], dtype=pl.Float64)

    assert_series_equal(distance_s, expected_s)
