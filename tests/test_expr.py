import numpy as np
import polars as pl
import shapely
from polars.testing import assert_frame_equal, assert_series_equal

import spatial_polars  # noqa: F401

from .fixtures import *  # NOQA:F403


def test_reproject(arch_mound_df: pl.DataFrame) -> None:
    """Test reprojections."""
    reprojected_shapely_array = (
        arch_mound_df.select(pl.col("geometry").spatial.reproject(32615))
        .to_series()
        .spatial.to_shapely_array()
    )

    expected_array = shapely.points(
        [(745063.298, 4278874.516), (755632.326, 4283223.681)],
    )

    geoms_match = shapely.equals_exact(reprojected_shapely_array, expected_array, 0.001)

    assert np.all(geoms_match)


def test_distance_scalar(two_points_df: pl.DataFrame) -> None:
    """Test distance scalar input."""
    distance_s = two_points_df.select(
        pl.col("geometry").spatial.distance(shapely.Point(0, 1)).alias("distance"),
    ).to_series()

    expected_s = pl.Series("distance", [1, 1], dtype=pl.Float64)

    assert_series_equal(distance_s, expected_s)


def test_distance_two_cols(
    two_points_df: pl.DataFrame,
    two_more_points_df: pl.DataFrame,
) -> None:
    """Test distance two column input."""
    df = pl.concat(
        [two_points_df, two_more_points_df.rename({"geometry": "other_geometry"})],
        how="horizontal",
    )

    distance_s = df.select(
        pl.struct(pl.col("geometry"), pl.col("other_geometry"))
        .spatial.distance()
        .alias("distance"),
    ).to_series()

    expected_s = pl.Series("distance", [10, 20], dtype=pl.Float64)

    assert_series_equal(distance_s, expected_s)


def test_dwithin_scalar_point_n_n(
    arch_mound_df: pl.DataFrame,
    busch_stadium_pitchers_mound_point: shapely.Point,
) -> None:
    """Test the dwithin with a point geometry."""
    expected_df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "dwithin_result": [False, False],
        },
    )
    result_df = arch_mound_df.select(
        pl.col("Place"),
        pl.col("geometry")
        .spatial.dwithin(
            busch_stadium_pitchers_mound_point,
            0.0000001,
        )
        .alias("dwithin_result"),
    )
    assert_frame_equal(result_df, expected_df)


def test_dwithin_scalar_point_y_n(
    arch_mound_df: pl.DataFrame,
    busch_stadium_pitchers_mound_point: shapely.Point,
) -> None:
    """Test the dwithin with a point geometry."""
    expected_df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "dwithin_result": [True, False],
        },
    )
    result_df = arch_mound_df.select(
        pl.col("Place"),
        pl.col("geometry")
        .spatial.dwithin(
            busch_stadium_pitchers_mound_point,
            0.01,
        )
        .alias("dwithin_result"),
    )
    assert_frame_equal(result_df, expected_df)


def test_dwithin_scalar_point_y_y(
    arch_mound_df: pl.DataFrame,
    busch_stadium_pitchers_mound_point: shapely.Point,
) -> None:
    """Test the dwithin with a point geometry."""
    expected_df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "dwithin_result": [True, True],
        },
    )
    result_df = arch_mound_df.select(
        pl.col("Place"),
        pl.col("geometry")
        .spatial.dwithin(
            busch_stadium_pitchers_mound_point,
            1,
        )
        .alias("dwithin_result"),
    )
    assert_frame_equal(result_df, expected_df)


def test_dwithin_scalar_polygon_n_n(
    arch_mound_df: pl.DataFrame,
    busch_stadium_pitchers_mound_point: shapely.Point,
) -> None:
    """Test the dwithin with a polygon geometry."""
    expected_df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "dwithin_result": [False, False],
        },
    )
    result_df = arch_mound_df.select(
        pl.col("Place"),
        pl.col("geometry")
        .spatial.dwithin(
            busch_stadium_pitchers_mound_point.buffer(0.00005),
            0.0000001,
        )
        .alias("dwithin_result"),
    )
    assert_frame_equal(result_df, expected_df)


def test_dwithin_scalar_polygon_y_n(
    arch_mound_df: pl.DataFrame,
    busch_stadium_pitchers_mound_point: shapely.Point,
) -> None:
    """Test the dwithin with a polygon geometry."""
    expected_df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "dwithin_result": [True, False],
        },
    )
    result_df = arch_mound_df.select(
        pl.col("Place"),
        pl.col("geometry")
        .spatial.dwithin(
            busch_stadium_pitchers_mound_point.buffer(0.00005),
            0.01,
        )
        .alias("dwithin_result"),
    )
    assert_frame_equal(result_df, expected_df)


def test_dwithin_scalar_polygon_y_y(
    arch_mound_df: pl.DataFrame,
    busch_stadium_pitchers_mound_point: shapely.Point,
) -> None:
    """Test the dwithin with a polygon geometry."""
    expected_df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "dwithin_result": [True, True],
        },
    )
    result_df = arch_mound_df.select(
        pl.col("Place"),
        pl.col("geometry")
        .spatial.dwithin(
            busch_stadium_pitchers_mound_point.buffer(0.00005),
            1,
        )
        .alias("dwithin_result"),
    )
    assert_frame_equal(result_df, expected_df)
