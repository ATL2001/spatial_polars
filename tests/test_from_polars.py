import polars as pl
import shapely
from polars.testing import assert_frame_equal

from spatial_polars import SpatialFrame

from .fixtures import *  # NOQA:F403


def test_from_point_coords(arch_mound_df: pl.DataFrame) -> None:
    """Test from point coordinates."""
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "x": [-90.18497, -90.06211],
            "y": [38.62456, 38.66072],
        },
    )
    s_df = SpatialFrame.from_point_coords(df, "x", "y")
    assert_frame_equal(s_df, arch_mound_df)


def test_from_WKB(arch_mound_df: pl.DataFrame) -> None:  # NOQA:N802
    """Test from wkt."""
    arch_wkb = shapely.Point(-90.18497, 38.62456).wkb
    monks_mound_wkb = shapely.Point(-90.06211, 38.66072).wkb
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "wkb": [arch_wkb, monks_mound_wkb],
        },
    )
    s_df = SpatialFrame.from_WKB(df, "wkb")
    assert_frame_equal(s_df, arch_mound_df.rename({"geometry": "wkb"}))


def test_from_WKB_named_geometry(arch_mound_df: pl.DataFrame) -> None:  # NOQA:N802
    """Test from wkb where input column is named 'geometry'."""
    arch_wkb = shapely.Point(-90.18497, 38.62456).wkb
    monks_mound_wkb = shapely.Point(-90.06211, 38.66072).wkb
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "geometry": [arch_wkb, monks_mound_wkb],
        },
    )
    s_df = SpatialFrame.from_WKB(df, "geometry")
    assert_frame_equal(s_df, arch_mound_df)


def test_from_WKT(arch_mound_df: pl.DataFrame) -> None:  # NOQA:N802
    """Test from wkt."""
    arch_wkt = shapely.Point(-90.18497, 38.62456).wkt
    monks_mound_wkt = shapely.Point(-90.06211, 38.66072).wkt
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "wkt": [arch_wkt, monks_mound_wkt],
        },
    )
    s_df = SpatialFrame.from_WKT(df, "wkt")
    assert_frame_equal(s_df, arch_mound_df.rename({"geometry": "wkt"}))


def test_from_WKT_named_geometry(arch_mound_df: pl.DataFrame) -> None:  # NOQA:N802
    """Test from wkt where input column is named 'geometry'."""
    arch_wkt = shapely.Point(-90.18497, 38.62456).wkt
    monks_mound_wkt = shapely.Point(-90.06211, 38.66072).wkt
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "geometry": [arch_wkt, monks_mound_wkt],
        },
    )
    s_df = SpatialFrame.from_WKT(df, "geometry")
    assert_frame_equal(s_df, arch_mound_df)
