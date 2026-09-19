import numpy as np
import polars as pl
import shapely
from polars.testing import assert_frame_equal

import spatial_polars  # noqa: F401

from .fixtures import *  # noqa: F403


def test_get_x(all_wkbs: list[bytes]) -> None:
    """Test wkb parsing/get_x expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_x(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_x(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_get_y(all_wkbs: list[bytes]) -> None:
    """Test wkb parsing/get_y expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_y(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_y(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_get_z(all_wkbs: list[bytes]) -> None:
    """Test wkb parsing/get_z expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_z(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_z(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_get_m(all_wkbs: list[bytes]) -> None:
    """Test wkb parsing/get_m expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_m(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_m(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_get_type_id(all_wkbs: list[bytes]) -> None:
    """Test the get_type_id expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_type_id(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_type_id(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_get_num_points(all_wkbs: list[bytes]) -> None:
    """Test the get_num_points expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_num_points(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_num_points(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_get_coordinate_dimension(
    all_wkbs: list[bytes],
) -> None:
    """Test get_coordinate_dimension expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.get_coordinate_dimension(shapely.from_wkb(wkb))
        result = (
            pl.DataFrame(
                {"wkb": wkb},
                schema={"wkb": pl.Binary},
            )
            .select(
                pl.col("wkb").spatial.from_WKB().spatial.get_coordinate_dimension(),
            )
            .item()
        )
        np.testing.assert_equal(result, expected)


def test_has_z(all_wkbs: list[bytes]) -> None:
    """Test has_z expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.has_z(shapely.from_wkb(wkb))

        result = pl.DataFrame(
            [[wkb]],
            orient="row",
            schema={"wkb": pl.Binary},
        ).select(
            pl.col("wkb").spatial.from_WKB().spatial.has_z().alias("has_z"),
        )["has_z"][0]
        assert result == expected


def test_has_m(all_wkbs: list[bytes]) -> None:
    """Test has_m expression."""
    for wkb in [*all_wkbs, None]:
        expected = shapely.has_m(shapely.from_wkb(wkb))

        result = pl.DataFrame(
            [[wkb]],
            orient="row",
            schema={"wkb": pl.Binary},
        ).select(
            pl.col("wkb").spatial.from_WKB().spatial.has_m().alias("has_m"),
        )["has_m"][0]
        assert result == expected
