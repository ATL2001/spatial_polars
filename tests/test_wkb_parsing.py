import polars as pl
from polars.testing import assert_frame_equal

import spatial_polars  # noqa: F401

from .fixtures import *  # noqa: F403


def test_points_get_x(all_points_wkb: list[bytes]) -> None:
    """Test point wkb parsing/get_x function for points."""
    expected_df = pl.DataFrame({"x_coord": 0}, schema={"x_coord": pl.Float64})
    for point_wkb in all_points_wkb:
        result = pl.DataFrame(
            {"geometry": point_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_x().alias("x_coord"),
        )

        assert_frame_equal(result, expected_df)


def test_points_get_y(all_points_wkb: list[bytes]) -> None:
    """Test point wkb parsing/get_y function for points."""
    expected_df = pl.DataFrame({"y_coord": 1}, schema={"y_coord": pl.Float64})
    for point_wkb in all_points_wkb:
        result = pl.DataFrame(
            {"geometry": point_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_y().alias("y_coord"),
        )

        assert_frame_equal(result, expected_df)


def test_points_get_z(z_points_wkb: list[bytes]) -> None:
    """Test point wkb parsing/get_z function for points."""
    expected_df = pl.DataFrame({"z_coord": 2}, schema={"z_coord": pl.Float64})
    for point_wkb in z_points_wkb:
        result = pl.DataFrame(
            {"geometry": point_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_z().alias("z_coord"),
        )
        assert_frame_equal(result, expected_df)


def test_points_get_m(m_points_wkb: list[bytes]) -> None:
    """Test point wkb parsing/get_m function for points."""
    expected_df = pl.DataFrame({"m_coord": 3}, schema={"m_coord": pl.Float64})
    for point_wkb in m_points_wkb:
        result = pl.DataFrame(
            {"geometry": point_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_m().alias("m_coord"),
        )
        assert_frame_equal(result, expected_df)


def test_non_points_get_xyzm(non_points_wkb: list[bytes]) -> None:
    """Test non-point wkb parsing/get_* function returns nan."""
    expected_df = pl.DataFrame(
        {
            "x_coord": float("nan"),
            "y_coord": float("nan"),
            "z_coord": float("nan"),
            "m_coord": float("nan"),
        },
        schema={
            "x_coord": pl.Float64,
            "y_coord": pl.Float64,
            "z_coord": pl.Float64,
            "m_coord": pl.Float64,
        },
    )
    for geom_wkb in non_points_wkb:
        result = pl.DataFrame(
            {"geometry": geom_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_m().alias("x_coord"),
            pl.col("geometry").spatial.from_WKB().spatial.get_m().alias("y_coord"),
            pl.col("geometry").spatial.from_WKB().spatial.get_m().alias("z_coord"),
            pl.col("geometry").spatial.from_WKB().spatial.get_m().alias("m_coord"),
        )
        assert_frame_equal(result, expected_df)


def test_non_z_points_get_z(non_z_points_wkb: list[bytes]) -> None:
    """Test point wkb parsing/get_z function for points without z."""
    expected_df = pl.DataFrame(
        {"z_coord": float("nan")},
        schema={"z_coord": pl.Float64},
    )
    for point_wkb in non_z_points_wkb:
        result = pl.DataFrame(
            {"geometry": point_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_z().alias("z_coord"),
        )
        assert_frame_equal(result, expected_df)


def test_non_m_points_get_m(non_m_points_wkb: list[bytes]) -> None:
    """Test point wkb parsing/get_m function for points without z."""
    expected_df = pl.DataFrame(
        {"m_coord": float("nan")},
        schema={"m_coord": pl.Float64},
    )
    for point_wkb in non_m_points_wkb:
        result = pl.DataFrame(
            {"geometry": point_wkb},
        ).select(
            pl.col("geometry").spatial.from_WKB().spatial.get_m().alias("m_coord"),
        )
        assert_frame_equal(result, expected_df)
