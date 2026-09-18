import polars as pl
import shapely
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


def test_get_type_id() -> None:
    """Test the get_type_id expression."""
    expected_df = pl.Series(
        values=[0, 0, 1, 1, 3, 3, 3, 4, 5, 6, 7, -1],
        name="type",
        dtype=pl.Int8,
    ).to_frame()

    point = shapely.Point(0, 0)
    point_z = shapely.Point(0, 0, 0)
    line_string = shapely.LineString([[0, 0], [1, 1]])
    line_string_z = shapely.LineString([[0, 0, 0], [1, 1, 1]])
    polygon = shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    polygon2 = shapely.Polygon([[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]])
    polygon_z = shapely.Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    multi_point = shapely.MultiPoint([(0.0, 0.0), (1.0, 1.0)])
    multi_line_string = shapely.MultiLineString([((0, 0), (1, 1)), ((-1, 0), (1, 0))])
    multi_polygon = shapely.MultiPolygon([polygon, polygon2])
    geometry_collection = shapely.GeometryCollection([point, polygon2])

    test_geoms = [
        point,
        point_z,
        line_string,
        line_string_z,
        polygon,
        polygon2,
        polygon_z,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
        None,
    ]

    wkbs = [[shapely.to_wkb(g, byte_order=1, flavor="iso")] for g in test_geoms]

    result = pl.DataFrame(wkbs, orient="row", schema={"wkb": pl.Binary}).select(
        pl.col("wkb")
        .spatial.from_WKB()
        .alias("geometry")
        .spatial.get_type_id()
        .alias("type"),
    )
    assert_frame_equal(result, expected_df)


def test_get_num_points() -> None:
    """Test the get_num_points expression."""
    expected_df = pl.Series(
        values=[2, 2, 3, 3, 0, 0, 0, 0],
        name="num_points",
        dtype=pl.UInt32,
    ).to_frame()

    two_point_line_string = shapely.LineString([[0, 0], [1, 1]])
    two_point_line_string_z = shapely.LineString([[0, 0, 0], [1, 1, 1]])
    three_point_line_string = shapely.LineString([[0, 0], [1, 1], [2, 0]])
    three_point_line_string_z = shapely.LineString([[0, 0, 0], [1, 1, 1], [2, 0, 0]])
    point = shapely.Point(0, 0)
    polygon = shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    multi_line_string = shapely.MultiLineString([((0, 0), (1, 1)), ((-1, 0), (1, 0))])

    test_geoms = [
        two_point_line_string,  # 2
        two_point_line_string_z,  # 2
        three_point_line_string,  # 3
        three_point_line_string_z,  # 3
        point,  # 0
        polygon,  # 0
        multi_line_string,  # 0
        None,  # 0
    ]

    wkbs = [[shapely.to_wkb(g, byte_order=1, flavor="iso")] for g in test_geoms]

    result = pl.DataFrame(wkbs, orient="row", schema={"wkb": pl.Binary}).select(
        pl.col("wkb")
        .spatial.from_WKB()
        .alias("geometry")
        .spatial.get_num_points()
        .alias("num_points"),
    )
    assert_frame_equal(result, expected_df)


def test_get_coordinate_dimension(
    little_endian_all_dimensions_points_wkb: list[bytes],
) -> None:
    """Test get_coordinate_dimension expression."""
    expected_df = pl.Series(
        values=[2, 3, 3, 4, -1],
        name="coordinate_dimension",
        dtype=pl.Int8,
    ).to_frame()

    result = pl.DataFrame(
        [[wkb] for wkb in [*little_endian_all_dimensions_points_wkb, None]],
        orient="row",
        schema={"wkb": pl.Binary},
    ).select(
        pl.col("wkb")
        .spatial.from_WKB()
        .spatial.get_coordinate_dimension()
        .alias("coordinate_dimension"),
    )
    assert_frame_equal(result, expected_df)
