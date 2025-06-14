import polars as pl
import pyproj
import pytest
import shapely


@pytest.fixture
def arch_mound_df() -> pl.DataFrame:
    """df with two rows of the name and geometry,
    one at the gateway arch one at monks mound wkid 4326."""
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
                pyproj.CRS.from_user_input(4326).to_wkt(), dtype=pl.Categorical
            ).alias("crs"),
        ).alias("geometry"),
    )


@pytest.fixture
def two_points_df() -> pl.DataFrame:
    """df with two rows of geometry, one at (0,0) one at (1,1) wkid 4326."""
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
                pyproj.CRS.from_user_input(4326).to_wkt(), dtype=pl.Categorical
            ).alias("crs"),
        ).alias("geometry"),
    )


@pytest.fixture
def two_more_points_df() -> pl.DataFrame:
    """df with two rows of geometry, one at (0, 10) one at (1, 21) wkid 4326."""
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
                pyproj.CRS.from_user_input(4326).to_wkt(), dtype=pl.Categorical
            ).alias("crs"),
        ).alias("geometry"),
    )
