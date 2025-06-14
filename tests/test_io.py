import datetime
import pathlib

import polars as pl
import pyproj
import pytest
import shapely
from polars.testing import assert_frame_equal

import spatial_polars
from spatial_polars import scan_spatial

test_dir = pathlib.Path(spatial_polars.__file__).parent.parent.parent / "tests"
test_data_dir = test_dir / "test_data"
test_data_gpkg = test_data_dir / "test_data.gpkg"


@pytest.fixture
def arch_mound_df() -> pl.DataFrame:
    return pl.DataFrame().with_columns(
        pl.Series("Place", ["Gateway Arch", "Monks Mound"], pl.String),
        pl.Series("f1_Int8", [0, 1], pl.Int8),
        pl.Series("f1_Int16", [0, 1], pl.Int16),
        pl.Series("f1_Int32", [0, 1], pl.Int32),
        pl.Series("f1_Int64", [0, 1], pl.Int64),
        pl.Series("f1_UInt8", [0, 1], pl.UInt8),
        pl.Series("f1_UInt16", [0, 1], pl.UInt16),
        pl.Series("f1_UInt32", [0, 1], pl.UInt32),
        pl.Series("f1_UInt64", [0, 1], pl.UInt64),
        pl.Series("f1_Boolean", [False, True], pl.Boolean),
        pl.Series("f1_Float32", [0, 1], pl.Float32),
        pl.Series("f1_Float64", [0, 1], pl.Float64),
        pl.Series(
            "f1_Date", [datetime.date(2000, 1, 1), datetime.date(2000, 12, 31)], pl.Date
        ),
        pl.Series(
            "f1_DT",
            [
                datetime.datetime(2000, 1, 1, 1, 1, 1),
                datetime.datetime(2000, 12, 31, 23, 59, 59),
            ],
            pl.Datetime("ms", time_zone="UTC"),
        ),
        pl.Series("f1_String", ["a", "z"], pl.String),
        pl.struct(
            pl.Series(
                "wkb_geometry",
                [
                    shapely.Point(-90.18497, 38.62456, 0).wkb,
                    shapely.Point(-90.06211, 38.66072, 1).wkb,
                ],
                dtype=pl.Binary,
            ),
            pl.lit(
                pyproj.CRS.from_user_input(4326).to_wkt(), dtype=pl.Categorical
            ).alias("crs"),
        ).alias("geometry"),
    )


# def test_scan_shp(arch_mound_df)->None:
#     # TODO, look into Z values and datetimes.
#     # lf = scan_spatial(test_data_dir / "shp" / "arch_mound.shp")
#     # assert_frame_equal(lf.collect(), arch_mound_df)
#     return

# def test_scan_fgb(arch_mound_df)->None:
#     # TODO, look into row ordering and datetimes.
#     # lf = scan_spatial(test_data_dir/ "arch_mound.fgb")
#     # assert_frame_equal(lf, arch_mound_df, check_dtypes=False)
#     return


def test_scan_geojson(arch_mound_df) -> None:
    lf = scan_spatial(test_data_dir / "arch_mound.geojson")
    assert_frame_equal(lf.collect(), arch_mound_df, check_dtypes=False)


def test_scan_geojsonsseq(arch_mound_df) -> None:
    lf = scan_spatial(test_data_dir / "arch_mound.geojsonl")
    assert_frame_equal(lf.collect(), arch_mound_df, check_dtypes=False)


def test_scan_gpkg(arch_mound_df) -> None:
    lf = scan_spatial(test_data_gpkg, layer="arch_mound")
    assert_frame_equal(lf.collect(), arch_mound_df, check_dtypes=False)


def test_scan_geoparquet(arch_mound_df) -> None:
    lf = scan_spatial(test_data_dir / "arch_mound.parquet")
    assert_frame_equal(lf.collect(), arch_mound_df)


def test_scan_parquet_bbox() -> None:
    # bbox will limit the arch_mound to just the arch row
    arch_bbox = (-90.19, 38.62, -90.07, 38.67)
    lf = scan_spatial(test_data_dir / "arch_mound.parquet", bbox=arch_bbox)
    assert lf.select(pl.len()).collect().item() == 1


def test_scan_geojson_bbox() -> None:
    # bbox will limit the arch_mound to just the arch row
    arch_bbox = (-90.19, 38.62, -90.07, 38.67)
    lf = scan_spatial(test_data_dir / "arch_mound.geojson", bbox=arch_bbox)
    assert lf.select(pl.len()).collect().item() == 1


def test_scan_parquet_mask() -> None:
    # mask will limit the arch_mound to just the arch row
    arch_bbox = (-90.19, 38.62, -90.07, 38.67)
    arch_mask = shapely.Polygon(shapely.box(*arch_bbox))
    lf = scan_spatial(test_data_dir / "arch_mound.parquet", mask=arch_mask)
    assert lf.select(pl.len()).collect().item() == 1


def test_scan_geojson_mask() -> None:
    # mask will limit the arch_mound to just the arch row
    arch_bbox = (-90.19, 38.62, -90.07, 38.67)
    arch_mask = shapely.Polygon(shapely.box(*arch_bbox))
    lf = scan_spatial(test_data_dir / "arch_mound.geojson", mask=arch_mask)
    assert lf.select(pl.len()).collect().item() == 1


def test_scan_subset_parquet_columns() -> None:
    # only selecting two columns should result in two columns being read
    lf = scan_spatial(test_data_dir / "arch_mound.parquet").select("geometry", "Place")
    assert len(lf.collect_schema()) == 2


def test_scan_subset_geojson_columns() -> None:
    # only selecting two columns should result in two columns being read
    lf = scan_spatial(test_data_dir / "arch_mound.geojson").select("geometry", "Place")
    assert len(lf.collect_schema()) == 2
