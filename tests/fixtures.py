import polars as pl
import pyproj
import pytest
import shapely


@pytest.fixture
def busch_stadium_pitchers_mound_point() -> shapely.Point:
    """Shapely Point located at the pitchers mound at Busch Stadium."""
    return shapely.Point(-90.19320, 38.62248)


@pytest.fixture
def arch_mound_df() -> pl.DataFrame:
    """Dataframe with two rows of the name and geometry.

    One at the gateway arch one at monks mound wkid 4326.
    """
    return pl.DataFrame().with_columns(
        pl.Series("Place", ["Gateway Arch", "Monks Mound"], pl.String),
        pl.struct(
            pl.Series(
                "wkb_geometry",
                [
                    shapely.Point(-90.18497, 38.62456).wkb,
                    shapely.Point(-90.06211, 38.66072).wkb,
                ],
                dtype=pl.Binary,
            ),
            pl.lit(
                pyproj.CRS.from_user_input(4326).to_wkt(),
                dtype=pl.Categorical,
            ).alias("crs"),
        ).alias("geometry"),
    )


@pytest.fixture
def two_points_df() -> pl.DataFrame:
    """Dataframe with two rows of geometry.

    One at (0,0) one at (1,1) wkid 4326.
    """
    return pl.DataFrame().with_columns(
        pl.struct(
            pl.Series(
                "wkb_geometry",
                [
                    shapely.Point(0, 0).wkb,
                    shapely.Point(1, 1).wkb,
                ],
                dtype=pl.Binary,
            ),
            pl.lit(
                pyproj.CRS.from_user_input(4326).to_wkt(),
                dtype=pl.Categorical,
            ).alias("crs"),
        ).alias("geometry"),
    )


@pytest.fixture
def two_more_points_df() -> pl.DataFrame:
    """Dataframe with two rows of geometry.

    One at (0, 10) one at (1, 21) wkid 4326.
    """
    return pl.DataFrame().with_columns(
        pl.struct(
            pl.Series(
                "wkb_geometry",
                [
                    shapely.Point(0, 10).wkb,
                    shapely.Point(1, 21).wkb,
                ],
                dtype=pl.Binary,
            ),
            pl.lit(
                pyproj.CRS.from_user_input(4326).to_wkt(),
                dtype=pl.Categorical,
            ).alias("crs"),
        ).alias("geometry"),
    )


# x =0 y=1 z=2 m=3
point_little_endian_iso_wkb = shapely.to_wkb(
    shapely.Point(0, 1),
    flavor="iso",
    byte_order=1,
)
point_little_endian_extended_wkb = shapely.to_wkb(
    shapely.Point(0, 1),
    flavor="extended",
    byte_order=1,
)

point_z_little_endian_iso_wkb = shapely.to_wkb(
    shapely.Point(0, 1, 2),
    flavor="iso",
    byte_order=1,
)
point_z_little_endian_extended_wkb = shapely.to_wkb(
    shapely.Point(0, 1, 2),
    flavor="extended",
    byte_order=1,
)

point_m_little_endian_iso_wkb = b"\x01\xd1\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@"  # noqa: E501
point_m_little_endian_extended_wkb = b"\x01\x01\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@"  # noqa: E501

point_zm_little_endian_iso_wkb = b"\x01\xb9\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@"  # noqa: E501
point_zm_little_endian_extended_wkb = b"\x01\x01\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@"  # noqa: E501

point_big_endian_iso_wkb = shapely.to_wkb(
    shapely.Point(0, 1),
    flavor="iso",
    byte_order=0,
)
point_big_endian_extended_wkb = shapely.to_wkb(
    shapely.Point(0, 1),
    flavor="extended",
    byte_order=0,
)

point_z_big_endian_iso_wkb = shapely.to_wkb(
    shapely.Point(0, 1, 2),
    flavor="iso",
    byte_order=0,
)
point_z_big_endian_extended_wkb = shapely.to_wkb(
    shapely.Point(0, 1, 2),
    flavor="extended",
    byte_order=0,
)

point_m_big_endian_iso_wkb = b"\x00\x00\x00\x07\xd1\x00\x00\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00@\x08\x00\x00\x00\x00\x00\x00"  # noqa: E501
point_m_big_endian_extended_wkb = b"\x00@\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00@\x08\x00\x00\x00\x00\x00\x00"  # noqa: E501

point_zm_big_endian_iso_wkb = b"\x00\x00\x00\x0b\xb9\x00\x00\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x08\x00\x00\x00\x00\x00\x00"  # noqa: E501
point_zm_big_endian_extended_wkb = b"\x00\xc0\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x08\x00\x00\x00\x00\x00\x00"  # noqa: E501


@pytest.fixture
def all_points_wkb() -> list[bytes]:
    """List of points wkb iso/extended big/little endian.

    xy, xyz, xym, xyzm coordinates are all included.
    """
    return [
        point_little_endian_iso_wkb,
        point_little_endian_extended_wkb,
        point_z_little_endian_iso_wkb,
        point_z_little_endian_extended_wkb,
        point_m_little_endian_iso_wkb,
        point_m_little_endian_extended_wkb,
        point_zm_little_endian_iso_wkb,
        point_zm_little_endian_extended_wkb,
        point_big_endian_iso_wkb,
        point_big_endian_extended_wkb,
        point_z_big_endian_iso_wkb,
        point_z_big_endian_extended_wkb,
        point_m_big_endian_iso_wkb,
        point_m_big_endian_extended_wkb,
        point_zm_big_endian_iso_wkb,
        point_zm_big_endian_extended_wkb,
    ]


@pytest.fixture
def z_points_wkb() -> list[bytes]:
    """List of points wkb iso/extended big/little endian.

    xyz, xyzm coordinates are all included.
    """
    return [
        point_z_little_endian_iso_wkb,
        point_z_little_endian_extended_wkb,
        point_zm_little_endian_iso_wkb,
        point_zm_little_endian_extended_wkb,
        point_z_big_endian_iso_wkb,
        point_z_big_endian_extended_wkb,
        point_zm_big_endian_iso_wkb,
        point_zm_big_endian_extended_wkb,
    ]


@pytest.fixture
def m_points_wkb() -> list[bytes]:
    """List of points wkb iso/extended big/little endian.

    xym, xyzm coordinates are all included.
    """
    return [
        point_m_little_endian_iso_wkb,
        point_m_little_endian_extended_wkb,
        point_zm_little_endian_iso_wkb,
        point_zm_little_endian_extended_wkb,
        point_m_big_endian_iso_wkb,
        point_m_big_endian_extended_wkb,
        point_zm_big_endian_iso_wkb,
        point_zm_big_endian_extended_wkb,
    ]


@pytest.fixture
def non_points_wkb() -> list[bytes]:
    """List of non-points wkb iso/extended big/little endian.

    xy, xyz coordinates are all included.

    Geometry types of LineString, LinearRing, MultiLineString, Polygon, MultiPolygon are
    included.
    """
    wkbs = []
    geoms = [
        shapely.LineString([(0, 0), (1, 1)]),
        shapely.LineString([(0, 0, 0), (1, 1, 1)]),
        shapely.LinearRing([(0, 0), (1, 1), (1, 0)]),
        shapely.LinearRing([(0, 0, 0), (1, 1, 1), (1, 0, -1)]),
        shapely.MultiLineString([((0, 0), (1, 1)), ((-1, 0), (1, 0))]),
        shapely.MultiLineString([((0, 0, 0), (1, 1, 1)), ((-1, 0, 1), (1, 0, -1))]),
        shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
        shapely.Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0]]),
        shapely.MultiPolygon(
            [
                shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                shapely.Polygon([[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]),
            ],
        ),
        shapely.MultiPolygon(
            [
                shapely.Polygon(
                    [[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5], [0, 0, 5]],
                ),
                shapely.Polygon(
                    [[10, 10, 5], [11, 10, 5], [11, 11, 5], [10, 11, 5], [10, 10, 5]],
                ),
            ],
        ),
    ]
    for byte_order in [1, 0]:
        for flavor in ["iso", "extended"]:
            for geom in geoms:
                wkbs.append(shapely.to_wkb(geom, byte_order=byte_order, flavor=flavor))  # noqa: PERF401

    return wkbs


@pytest.fixture
def non_z_points_wkb() -> list[bytes]:
    """List of points wkb iso/extended big/little endian.

    xy, xym coordinates are all included.
    """
    return [
        point_little_endian_iso_wkb,
        point_little_endian_extended_wkb,
        point_m_little_endian_iso_wkb,
        point_m_little_endian_extended_wkb,
        point_big_endian_iso_wkb,
        point_big_endian_extended_wkb,
        point_m_big_endian_iso_wkb,
        point_m_big_endian_extended_wkb,
    ]


@pytest.fixture
def non_m_points_wkb() -> list[bytes]:
    """List of points wkb iso/extended big/little endian.

    xy, xyz coordinates are all included.
    """
    return [
        point_little_endian_iso_wkb,
        point_little_endian_extended_wkb,
        point_z_little_endian_iso_wkb,
        point_z_little_endian_extended_wkb,
        point_big_endian_iso_wkb,
        point_big_endian_extended_wkb,
        point_z_big_endian_iso_wkb,
        point_z_big_endian_extended_wkb,
    ]
