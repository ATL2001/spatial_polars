"""Spatial Polars SpatialLazyFrame.

This module provides a `SpatialLazyFrame` class which enables a "spatial" namespace on
polars lazyframes.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
import shapely
from polars import col as c

try:
    from scipy.spatial import KDTree

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False


@pl.api.register_lazyframe_namespace("spatial")
class SpatialLazyFrame:
    """Spatial Polars Spatial lazyframe."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        """For making polars do spatial stuff."""
        self._lf = lf

    def join(
        self,
        other: pl.LazyFrame,
        how: Literal["left", "right", "full", "inner", "semi", "anti"] = "inner",
        predicate: Literal[
            "intersects",
            "within",
            "contains",
            "overlaps",
            "crosses",
            "touches",
            "covers",
            "covered_by",
            "contains_properly",
            "dwithin",
        ] = "intersects",
        distance: float | None = None,
        on: str = "geometry",
        left_on: str | None = None,
        right_on: str | None = None,
        suffix: str = "_right",
        maintain_order: Literal[
            "none",
            "left",
            "right",
            "left_right",
            "right_left",
        ] = "none",
    ) -> pl.LazyFrame:
        r"""Join two SpatialLazyFrames based on a spatial predicate.

        Parameters
        ----------
        other
            SpatialLazyFrames to join with.

        how
            Join strategy.

            * *inner*
                Returns rows that have matching values in both tables
            * *left*
                Returns all rows from the left table, and the matched rows from the
                right table
            * *right*
                Returns all rows from the right table, and the matched rows from the
                left table
            * *full*
                Returns all rows when there is a match in either left or right table
            * *semi*
                Returns rows from the left table that have a match in the right table.
            * *anti*
                Returns rows from the left table that have no match in the right table.

        predicate
            The predicate to use for testing geometries from the tree that are within
            the input geometry's bounding box.
            * *intersects*
                Joins rows in the left frame to the right frame if they share any
                portion of space.

            * *within*
                Joins rows in the left frame to the right if they are completely inside
                a geometry from the right frame.

            * *contains*
                Joins rows in the left frame to the right if the geometry from the right
                frame is completely inside the geometry from the left frame

            * *overlaps*
                Joins rows in the left frame to the right if they have some but not all
                points/space in common, have the same dimension, and the intersection of
                the interiors of the two geometries has the same dimension as the
                geometries themselves.

            * *crosses*
                Joins rows in the left frame to the right if they have some but not all
                interior points in common, the intersection is one dimension less than
                the maximum dimension for the geomtries.

            * *touches*
                Joins rows in the left frame to the right if they only share points on
                their boundaries.

            * *covers*
                Joins rows in the left frame to the right if no point of the right
                geometry is outside of the left geometry.


            * *covered_by*
                Joins rows in the left frame to the right if no point of the left
                geometry is outside of the right geometry.


            * *contains_properly*
                Joins rows in the left frame to the right if the geometry from the right
                is completely inside the geometry from the left with no common boundary
                points.


            * *dwithin*
                Joins rows in the left frame to the right if they are within the given
                `distance` of one another.

        distance
            Distances around each input geometry to join for the `dwithin` predicate.
            Required if predicate=`dwithin`.

        on
            Name of the geometry columns in both SpatialLazyFrames.

        left_on
            Name of the geometry column in the left SpatialLazyFrames for the spatial join.

        right_on
            Name of the geometry column in the right SpatialLazyFrames for the spatial join.

        suffix
            Suffix to append to columns with a duplicate name.

        maintain_order
            Which LazyFrame row order to preserve, if any.
            Do not rely on any observed ordering without explicitly
            setting this parameter, as your code may break in a future release.
            Not specifying any ordering can improve performance
            Supported for inner, left, right and full joins

            * *none*
                No specific ordering is desired. The ordering might differ across
                Polars versions or even between different runs.
            * *left*
                Preserves the order of the left LazyFrame.
            * *right*
                Preserves the order of the right LazyFrame.
            * *left_right*
                First preserves the order of the left LazyFrame, then the right.
            * *right_left*
                First preserves the order of the right LazyFrame, then the left.

        Note
        ----
        Spatial joins only take into account x/y coodrdinates, any Z values present in
        the geometries are ignored.

        Examples
        --------
        **Spatial join roads that intersect rails**

        >>> import polars as pl
        >>> from spatial_polars import scan_spatial
        >>> zipped_data = r"C:\data\illinois-latest-free.shp.zip"
        >>> roads_df, rails_df = pl.collect_all([
        >>>         scan_spatial(zipped_data, "gis_osm_roads_free_1").select("name", "geometry"),
        >>>         scan_spatial(zipped_data, "gis_osm_railways_free_1").select("name", "geometry")
        >>>     ],
        >>>     engine="streaming"
        >>> )
        >>> roads_rails_df = roads_df.spatial.join(
        >>>     rails_df,
        >>>     suffix="_rail"
        >>> )
        >>> roads_rails_df
        shape: (43_772, 4)
        ┌─────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
        │ name            ┆ geometry                 ┆ name_rail                ┆ geometry_rail            │
        │ ---             ┆ ---                      ┆ ---                      ┆ ---                      │
        │ str             ┆ struct[2]                ┆ str                      ┆ struct[2]                │
        ╞═════════════════╪══════════════════════════╪══════════════════════════╪══════════════════════════╡
        │ Kingery Highway ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00Y │
        │                 ┆ x02\x0…                  ┆                          ┆ \x00\x…                  │
        │ Kingery Highway ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00] │
        │                 ┆ x02\x0…                  ┆                          ┆ \x00\x…                  │
        │ Kingery Highway ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00[ │
        │                 ┆ x02\x0…                  ┆                          ┆ \x00\x…                  │
        │ Kingery Highway ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00Y │
        │                 ┆ x02\x0…                  ┆                          ┆ \x00\x…                  │
        │ Kingery Highway ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00] │
        │                 ┆ x02\x0…                  ┆                          ┆ \x00\x…                  │
        │ …               ┆ …                        ┆ …                        ┆ …                        │
        │ null            ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00\ │
        │                 ┆ x02\x0…                  ┆                          ┆ x02\x0…                  │
        │ null            ┆ {b"\x01\x02\x00\x00\x00\ ┆ BNSF Chicago Subdivision ┆ {b"\x01\x02\x00\x00\x00\ │
        │                 ┆ x02\x0…                  ┆                          ┆ x02\x0…                  │
        │ null            ┆ {b"\x01\x02\x00\x00\x00\ ┆ UP Kenosha Subdivision   ┆ {b"\x01\x02\x00\x00\x00\ │
        │                 ┆ x02\x0…                  ┆                          ┆ x02\x0…                  │
        │ null            ┆ {b"\x01\x02\x00\x00\x00\ ┆ UP Kenosha Subdivision   ┆ {b"\x01\x02\x00\x00\x00\ │
        │                 ┆ x02\x0…                  ┆                          ┆ x02\x0…                  │
        │ null            ┆ {b"\x01\x02\x00\x00\x00\ ┆ Matteson Subdivision     ┆ {b"\x01\x02\x00\x00\x00\ │
        │                 ┆ x16\x0…                  ┆                          ┆ x1f\x0…                  │
        └─────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘

        """  # NOQA:E501
        if left_on is None:
            left_on = on
        if right_on is None:
            right_on = on

        def _make_tree_df() -> pl.DataFrame:
            self_geometries_df = self._lf.select(left_on).collect(engine="in-memory")
            other_geometries_df = other.select(right_on).collect(engine="in-memory")

            self_geometries = self_geometries_df[left_on].spatial.to_shapely_array()
            other_geometries = other_geometries_df[right_on].spatial.to_shapely_array()
            return pl.DataFrame(
                shapely.STRtree(other_geometries)
                .query(self_geometries, predicate=predicate, distance=distance)
                .T,
                schema={"left_index": pl.Int64, "right_index": pl.Int64},
            )

        tree_query_df = pl.defer(
            _make_tree_df,
            schema={"left_index": pl.Int64, "right_index": pl.Int64},
            validate_schema=False,
        )


        if how in ["left", "right", "full", "inner"]:
            joined = (
                self._lf.with_row_index("left_index")
                .join(
                    tree_query_df,
                    how=how,
                    on="left_index",
                    maintain_order=maintain_order,
                )
                .join(
                    other.with_row_index("right_index"),
                    how=how,
                    on="right_index",
                    suffix=suffix,
                    maintain_order=maintain_order,
                )
                .drop("right_index", "left_index")
            )
        elif how in ["semi", "anti"]:
            joined = (
                self._lf.with_row_index("left_index")
                .join(
                    tree_query_df,
                    how=how,
                    on="left_index",
                    maintain_order=maintain_order,
                )
                .drop(c.left_index)
            )

        return joined

    def join_nearest(
        self,
        other: pl.LazyFrame,
        max_distance: float | None = None,
        on: str = "geometry",
        left_on: str | None = None,
        right_on: str | None = None,
        suffix: str = "_right",
        maintain_order: Literal[
            "none",
            "left",
            "right",
            "left_right",
            "right_left",
        ] = "none",
        *,
        return_distance: bool = False,
        exclusive: bool = False,
        all_matches: bool = True,
    ) -> pl.LazyFrame:
        r"""Join two SpatialLazyFrames based on a spatial distance.

        Parameters
        ----------
        other
            SpatialLazyFrames to join with.

        max_distance
            The maximum distance to search around an input feature.

        on
            Name of the geometry columns in both SpatialLazyFrames.

        left_on
            Name of the geometry column in the left SpatialLazyFrames for the spatial
            join.

        right_on
            Name of the geometry column in the right SpatialLazyFrames for the spatial
            join.

        suffix
            Suffix to append to columns with a duplicate name.

        maintain_order
            Which LazyFrame row order to preserve, if any.
            Do not rely on any observed ordering without explicitly
            setting this parameter, as your code may break in a future release.
            Not specifying any ordering can improve performance
            Supported for inner, left, right and full joins

            * *none*
                No specific ordering is desired. The ordering might differ across
                Polars versions or even between different runs.
            * *left*
                Preserves the order of the left LazyFrame.
            * *right*
                Preserves the order of the right LazyFrame.
            * *left_right*
                First preserves the order of the left LazyFrame, then the right.
            * *right_left*
                First preserves the order of the right LazyFrame, then the left.

        return_distance
            If True, will return distances between joined features.

        exclusive
            If True, geometries that are equal to the input geometry will not be
            returned.

        all_matches
            If True, all equidistant and intersected geometries will be returned for
            each input geometry. If False, only the first nearest geometry will be
            returned.

        Note
        ----
        Spatial joins only take into account x/y coodrdinates, any Z values present in
        the geometries are ignored.

        """
        if left_on is None:
            left_on = on
        if right_on is None:
            right_on = on

        def _make_tree_df() -> pl.DataFrame:
            self_geometries_df = self._lf.select(left_on).collect(engine="in-memory")
            other_geometries_df = other.select(right_on).collect(engine="in-memory")

            self_geometries = self_geometries_df[left_on].spatial.to_shapely_array()
            other_geometries = other_geometries_df[right_on].spatial.to_shapely_array()

            query_results = shapely.STRtree(self_geometries).query_nearest(
                other_geometries,
                max_distance=max_distance,
                return_distance=return_distance,
                exclusive=exclusive,
                all_matches=all_matches,
            )

            if return_distance is True:
                return pl.DataFrame(
                    query_results[0].T,
                    schema={"right_index": pl.Int64, "left_index": pl.Int64},
                ).with_columns(pl.Series("distance", query_results[1]))

            return pl.DataFrame(
                query_results,
                schema={"right_index": pl.Int64, "left_index": pl.Int64},
            )

        tree_query_df = pl.defer(
            _make_tree_df,
            schema={"left_index": pl.Int64, "right_index": pl.Int64},
            validate_schema=False,
        )

        return (
            self._lf.with_row_index("left_index")
            .join(
                tree_query_df,
                how="left",
                on="left_index",
                maintain_order=maintain_order,
            )
            .join(
                other.with_row_index("right_index"),
                how="left",
                on="right_index",
                suffix=suffix,
                maintain_order=maintain_order,
            )
            .drop("right_index", "left_index")
        )

    def centroid_knn_join(
        self,
        other: pl.LazyFrame,
        k: int,
        on: str = "geometry",
        left_on: str | None = None,
        right_on: str | None = None,
        suffix: str = "_right",
        *,
        left_all_points:bool = False,
        right_all_points:bool = False,
    ) -> pl.LazyFrame:
        r"""Perform K nearest neighbors join of centroids of geometries in two frames.

        Parameters
        ----------
        other
            SpatialFrame to join with.

        k
            The number of nearest neighbors to include.

        on
            Name of the geometry columns in both SpatialFrames.

        left_on
            Name of the geometry column in the left SpatialFrame for the spatial join.

        right_on
            Name of the geometry column in the right SpatialFrame for the spatial join.

        suffix
            Suffix to append to columns with a duplicate name.

        left_all_points
            If the left geometries are already points, setting this to `True` will skip
            a step of computing the geometry's centroid.

        right_all_points
            If the right geometries are already points, setting this to `True` will skip
            a step of computing the geometry's centroid.

        Notes
        -----
            As the name implies, this KNN join method only takes into account the
            centroids of the geometries in both LazyFrames, it may not be suitable
            for joining the nearest lines or polygons depending on the distribution of
            the geometries.

            This method relies on scipy.spatial's KDTree to find the neighbors.

        """
        if left_on is None:
            left_on = on
        if right_on is None:
            right_on = on

        self_lf = self._lf

        def _make_tree_df() -> pl.DataFrame:
            if left_all_points:
                self_centroids_df = self_lf.select(
                    pl.col(left_on),
                ).collect(engine="in-memory")
            else:
                self_centroids_df = self_lf.select(
                    pl.col(left_on).spatial.centroid(),
                ).collect(engine="in-memory")

            if right_all_points:
                other_centroids_df = other.select(
                    pl.col(right_on),
                ).collect(engine="in-memory")
            else:
                other_centroids_df = other.select(
                    pl.col(right_on).spatial.centroid(),
                ).collect(engine="in-memory")

            self_centroids = self_centroids_df[left_on].spatial.to_shapely_array()
            other_centroids = other_centroids_df[right_on].spatial.to_shapely_array()

            self_coords = shapely.get_coordinates(self_centroids)
            other_coords = shapely.get_coordinates(other_centroids)

            tree = KDTree(other_coords)
            query_result = tree.query(self_coords, k=k)
            return pl.DataFrame(query_result[1])

        tree_query_df = pl.defer(
            _make_tree_df,
            schema={f"column_{i}": pl.Int64 for i in range(k)},
            validate_schema=False,
        )

        return (
            tree_query_df.lazy()
            .with_row_index("self_index")
            .unpivot(
                index="self_index",
                value_name="other_index",
            )
            .drop(
                "variable",
            )
            .join(
                self_lf.with_row_index("self_index"),
                how="left",
                on="self_index",
            )
            .join(
                other.with_row_index("other_index"),
                how="left",
                on="other_index",
                suffix=suffix,
            )
        )
