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

@pytest.fixture
def all_wkbs() -> list[bytes]:
    """List of all wkb possibilites.

    144 different WKB entries.

    ### Includes WKB for:
        #### Geometry Types:

            * point
            * linestring
            * polygon
            * multipoint
            * multilinestring
            * multipolygon

        #### Coordinates:

            * xy
            * xyz
            * xym
            * xyzm

        #### Byte order:

            * little endian
            * big endian

        #### WKB flavor:

            * ISO
            * Extended
            * Extended with embedded SRID

    """
    wkbs = []
    geoms = [
        shapely.Point(0, 1),
        shapely.Point(0, 1, 2),
        shapely.MultiPoint([(0, 1), (2, 3)]),
        shapely.MultiPoint([(0, 1, 2), (3, 4, 5)]),
        shapely.LineString([(0, 1), (2, 3)]),
        shapely.LineString([(0, 1, 2), (3, 4, 5)]),
        shapely.MultiLineString([((0, 1), (2, 3)), ((4, 5), (6, 7))]),
        shapely.MultiLineString([((0, 1, 2), (3, 4, 5)), ((6, 7, 8), (9, 10, 11))]),
        shapely.Polygon([[0, 0], [1, 2], [3, 4], [5, 6], [0, 0]]),
        shapely.Polygon([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]),
        shapely.MultiPolygon(
            [
                shapely.Polygon([[0, 0], [1, 2], [3, 4], [5, 6], [0, 0]]),
                shapely.Polygon([[10, 10], [11, 12], [13, 14], [15, 16], [10, 10]]),
            ],
        ),
        shapely.MultiPolygon(
            [
                shapely.Polygon(
                    [[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
                ),
                shapely.Polygon(
                    [[10, 10, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [10, 10, 10]],
                ),
            ],
        ),
    ]

    m_geoms = []
    for geom in geoms:
        # replace Z coordinates with M
        wkt = shapely.to_wkt(geom)
        if "Z" in wkt:
            m_geoms.append(shapely.from_wkt(wkt.replace("Z","M")))

    # add some ZM geometries
    m_geoms.extend([shapely.from_wkt(wkt) for wkt in [
        "POINT ZM (0 1 2 3)",
        "MULTIPOINT ZM ((0 1 2 3), (4 5 6 7))",
        "LINESTRING ZM (0 1 2 6, 3 4 5 7)",
        "MULTILINESTRING ZM ((0 1 2 12, 3 4 5 13), (6 7 8 14, 9 10 11 15))",
        "POLYGON ZM ((0 0 0 0, 1 2 3 10, 4 5 6 11, 7 8 9 12, 0 0 0 0))",
        "MULTIPOLYGON ZM (((0 0 0 0, 1 2 3 10, 4 5 6 11, 7 8 9 12, 0 0 0 0)), ((10 10 10 10, 11 12 13 20, 14 15 16 21, 17 18 19 22, 10 10 10 10)))",
    ]])
    geoms.extend(m_geoms)
    geoms.sort(key=shapely.to_wkt)

    for byte_order in [1, 0]:
        for geom in geoms:
            wkbs.append(shapely.to_wkb(geom, byte_order=byte_order, flavor="iso"))
            wkbs.append(shapely.to_wkb(geom, byte_order=byte_order, flavor="extended"))
            wkbs.append(
                shapely.to_wkb(
                    shapely.set_srid(geom, 4326),
                    byte_order=byte_order,
                    flavor="extended",
                ),
            )
    return wkbs
