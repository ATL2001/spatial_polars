import polars as pl
import shapely
from fixtures import *  # NOQA F403
from polars.testing import assert_frame_equal

from spatial_polars import SpatialFrame


def test_from_point_coords(arch_mound_df: pl.DataFrame) -> None:
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "x": [-90.18497, -90.06211],
            "y": [38.62456, 38.66072],
        }
    )
    s_df = SpatialFrame.from_point_coords(df, "x", "y")
    assert_frame_equal(s_df, arch_mound_df)


def test_from_WKB(arch_mound_df: pl.DataFrame) -> None:
    arch_wkb = shapely.Point(-90.18497, 38.62456).wkb
    monks_mound_wkb = shapely.Point(-90.06211, 38.66072).wkb
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "wkb": [arch_wkb, monks_mound_wkb],
        }
    )
    s_df = SpatialFrame.from_WKB(df, "wkb")
    assert_frame_equal(s_df, arch_mound_df)


def test_from_WKT(arch_mound_df: pl.DataFrame) -> None:
    arch_wkt = shapely.Point(-90.18497, 38.62456).wkt
    monks_mound_wkt = shapely.Point(-90.06211, 38.66072).wkt
    df = pl.DataFrame(
        {
            "Place": ["Gateway Arch", "Monks Mound"],
            "wkt": [arch_wkt, monks_mound_wkt],
        }
    )
    s_df = SpatialFrame.from_WKT(df, "wkt")
    assert_frame_equal(s_df, arch_mound_df)
