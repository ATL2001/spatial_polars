"""Spatial Polars SpatialFrame.

This module provides a `SpatialFrame` class which enables a "spatial" namespace on
polars dataframes.
"""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING, Any, Literal

import geoarrow.pyarrow as ga
import polars as pl
import pyogrio
import pyproj
import shapely
from geoarrow.pyarrow import io as gaio
from polars import col as c

from ._utils import validate_cmap_input, validate_width_and_radius_input

if TYPE_CHECKING:
    from io import BytesIO

    import pyarrow as pa
    from lonboard import Map
    from lonboard.types.layer import (
        PathLayerKwargs,
        PolygonLayerKwargs,
        ScatterplotLayerKwargs,
    )
    from lonboard.types.map import MapKwargs
    from matplotlib.colors import Colormap
    from numpy import floating
    from numpy.typing import NDArray
    from palettable.palette import Palette


__all__ = [
    "SpatialFrame",
]

try:
    from lonboard import PathLayer, PolygonLayer, ScatterplotLayer, viz
    from lonboard.colormap import apply_categorical_cmap, apply_continuous_cmap

    HAS_LONBOARD = True
except ModuleNotFoundError:
    HAS_LONBOARD = False

try:
    from scipy.spatial import KDTree

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False


def check_lonboard() -> None:
    if HAS_LONBOARD is False:
        err = "Lonboard not installed, to use this function, please install lonboard."
        raise ModuleNotFoundError(err)


def check_scipy() -> None:
    if HAS_SCIPY is False:
        err = "Scipy not installed, to use this function, please install scipy."
        raise ModuleNotFoundError(err)


@pl.api.register_dataframe_namespace("spatial")
class SpatialFrame:
    """Spatial Polars Spatial frame."""

    def __init__(self, df: pl.DataFrame) -> None:
        """For making polars do spatial stuff."""
        self._df = df

    def write_spatial(
        self,
        path: str | BytesIO,
        layer: str | None = None,
        *,
        driver: str | None = None,
        geometry_name: str = "geometry",
        geometry_type: str | None = None,
        encoding: str | None = None,
        append: bool = False,
        dataset_metadata: dict | None = None,
        layer_metadata: dict | None = None,
        metadata: dict | None = None,
        dataset_options: dict | None = None,
        layer_options: dict | None = None,
    ) -> None:
        r"""Write the dataframe to a format supported by [pyogrio](https://pyogrio.readthedocs.io/en/latest/supported_formats.html).

        Parameters
        ----------
        path
            path to output file on writeable file system or an io.BytesIO object to allow writing to memory NOTE: support for writing to memory is limited to specific drivers.

        layer
            layer name to create. If writing to memory and layer name is not provided, it layer name will be set to a UUID4 value.

        driver
            The OGR format driver used to write the vector file. By default attempts to infer driver from path. Must be provided to write to memory.

        geometry_name
            The name of the column in the dataframe that will be written as the geometry field.

        geometry_type
            The geometry type of the written layer. Currently, this needs to be specified explicitly when creating a new layer with geometries. Possible values are: “Unknown”, “Point”, “LineString”, “Polygon”, “MultiPoint”, “MultiLineString”, “MultiPolygon” “GeometryCollection”, “Point Z”, “LineString Z”, “Polygon Z”, “MultiPoint Z”, “MultiLineString Z”, “MultiPolygon Z” or “GeometryCollection Z”.

            This parameter does not modify the geometry, but it will try to force the layer type of the output file to this value. Use this parameter with caution because using a wrong layer geometry type may result in errors when writing the file, may be ignored by the driver, or may result in invalid files.

        encoding
            Only used for the .dbf file of ESRI Shapefiles. If not specified, uses the default locale.

        append
            If True, the data source specified by path already exists, and the driver supports appending to an existing data source, will cause the data to be appended to the existing records in the data source. Not supported for writing to in-memory files. NOTE: append support is limited to specific drivers and GDAL versions.

        dataset_metadata
            Metadata to be stored at the dataset level in the output file; limited to drivers that support writing metadata, such as GPKG, and silently ignored otherwise. Keys and values must be strings.

        layer_metadata
            Metadata to be stored at the layer level in the output file; limited to drivers that support writing metadata, such as GPKG, and silently ignored otherwise. Keys and values must be strings.

        metadata
            alias of layer_metadata.

        dataset_options
            Dataset creation options (format specific) passed to OGR. Specify as a key-value dictionary.

        layer_options
            Layer creation options (format specific) passed to OGR. Specify as a key-value dictionary.

        Examples
        --------
        **Writing a shapefile**
        >>> from spatial_polars import read_spatial
        >>> my_shapefile = r"c:\data\roads.shp"
        >>> df = read_spatial(my_shapefile)
        >>> df.spatial.write_spatial(r"C:\data\roads_2.shp")

        **Writing a geopackage**
        >>> df.spatial.write_spatial(r"C:\random_data\my_geopackage.gpkg", layer="roads")

        """  # NOQA: E501
        geometries_wkb = (
            self._df[geometry_name].struct.field("wkb_geometry").to_numpy().copy()
        )
        pa_table = (
            self._df.drop(geometry_name)
            .with_columns(pl.Series(geometry_name, geometries_wkb, dtype=pl.Binary))
            .to_arrow()
        )
        if geometry_type is None:
            geom_wkb = geometries_wkb[0]
            geom = shapely.from_wkb(geom_wkb)
            geometry_type = geom.geom_type

        crs = self._df[geometry_name].struct.field("crs")[0]
        pyogrio.write_arrow(
            pa_table,
            path=path,
            layer=layer,
            driver=driver,
            geometry_name=geometry_name,
            geometry_type=geometry_type,
            crs=crs,
            encoding=encoding,
            append=append,
            dataset_metadata=dataset_metadata,
            layer_metadata=layer_metadata,
            metadata=metadata,
            dataset_options=dataset_options,
            layer_options=layer_options,
        )

    def write_geoparquet(
        self,
        path: str,
        geometry_name: str = "geometry",
        *,
        write_bbox: bool = False,
        write_geometry_types: bool | None = None,
    ) -> None:
        r"""Write the dataframe to a geoparquet file.

        Parameters
        ----------
        path
            path to output file on writeable file system.

        geometry_name
            The name(s) of the column(s) in the dataframe that will be written with
            geoarrow metadata.

        write_bbox
            May be computationally expensive for large input.

        write_geometry_types
            May be computationally expensive for large input.

        Note
        ----
        Any rows with null geometries will be discarded.

        Examples
        --------
        >>> from spatial_polars import read_spatial
        >>> my_shapefile = r"c:\data\roads.shp"
        >>> df = read_spatial(my_shapefile)
        >>> df.spatial.write_geoparquet(r"c:\data\roads.parquet")

        """
        geoarrow_table = self.to_geoarrow(geometry_name)
        gaio.write_geoparquet_table(
            geoarrow_table,
            path,
            write_bbox=write_bbox,
            write_geometry_types=write_geometry_types,
        )

    def to_geoarrow(
        self,
        geometry_name: list[str] | str = "geometry",
    ) -> pa.Array:
        r"""Convert the dataframe to geoarrow table.

        Parameters
        ----------
        geometry_name
            The name(s) of the column(s) in the dataframe that will be written with
            geoarrow metadata.

        Note
        ----
        Any rows with null geometries will be discarded.


        Examples
        --------
        >>> from spatial_polars import read_spatial
        >>> my_shapefile = r"c:\data\roads.shp"
        >>> df = read_spatial(my_shapefile)
        >>> df.spatial.to_geoarrow()
        pyarrow.Table
        osm_id: large_string
        code: int32
        fclass: large_string
        name: large_string
        ref: large_string
        oneway: large_string
        maxspeed: int32
        layer: int64
        bridge: large_string
        tunnel: large_string
        geometry: extension<geoarrow.linestring<LinestringType>>
        osm_id: [["4265057","4265058","4267607","4271616","4275365",...,"4372351","4372353","4374903","4374905","4374906"],["4375793","4376011","4377106","4377123","4377209",...,"4493766","4493790","4500373","4500375","4516633"],...,["1370367863","1370367864","1370367868","1370367873","1370367874",...,"1370383552","1370383553","1370383554","1370383556","1370383557"],["1370383558","1370383559","1370383560","1370383561","1370383562",...,"1370383592","1370383593","1370383594","1370383595","1370398885"]]
        code: [[5114,5114,5114,5115,5122,...,5122,5152,5141,5122,5141],[5111,5111,5131,5131,5115,...,5114,5111,5152,5152,5111],...,[5153,5153,5153,5153,5153,...,5153,5153,5153,5141,5141],[5141,5153,5153,5153,5153,...,5153,5153,5153,5153,5141]]
        fclass: [["secondary","secondary","secondary","tertiary","residential",...,"residential","cycleway","service","residential","service"],["motorway","motorway","motorway_link","motorway_link","tertiary",...,"secondary","motorway","cycleway","cycleway","motorway"],...,["footway","footway","footway","footway","footway",...,"footway","footway","footway","service","service"],["service","footway","footway","footway","footway",...,"footway","footway","footway","footway","service"]]
        name: [["55th Street","Fairview Avenue","31st Street","59th Street","61st Street",...,"Fairmount Avenue",null,null,"Mochel Drive",null],["Kennedy Expressway","Kennedy Expressway",null,null,"59th Street",...,"Midwest Road","Ronald Reagan Memorial Tollway","Main Trail",null,"Borman Expressway"],...,[null,null,null,null,null,...,null,null,null,null,null],[null,null,null,null,null,...,null,null,null,null,null]]
        ref: [[null,null,null,null,null,...,null,null,null,null,null],["I 190","I 190",null,null,null,...,null,"I 88;IL 110",null,null,"I 80;I 94;US 6"],...,[null,null,null,null,null,...,null,null,null,null,null],[null,null,null,null,null,...,null,null,null,null,null]]
        oneway: [["F","B","B","B","B",...,"B","B","B","F","F"],["F","F","F","F","B",...,"B","F","B","B","F"],...,["B","B","B","B","B",...,"B","B","B","B","B"],["B","B","B","B","B",...,"B","B","B","B","B"]]
        maxspeed: [[0,0,72,0,0,...,0,0,0,0,0],[0,0,0,0,0,...,0,96,0,0,88],...,[0,0,0,0,0,...,0,0,0,0,0],[0,0,0,0,0,...,0,0,0,0,0]]
        layer: [[0,0,0,0,0,...,0,0,0,0,0],[0,0,0,0,0,...,0,0,0,0,0],...,[0,0,0,0,0,...,0,0,0,0,0],[0,0,0,0,0,...,0,0,0,0,0]]
        bridge: [["F","F","F","F","F",...,"F","F","F","F","F"],["F","F","F","F","F",...,"F","F","F","F","F"],...,["F","F","F","F","F",...,"F","F","F","F","F"],["F","F","F","F","F",...,"F","F","F","F","F"]]
        tunnel: [["F","F","F","F","F",...,"F","F","F","F","F"],["F","F","F","F","F",...,"F","F","F","F","F"],...,["F","F","F","F","F",...,"F","F","F","F","F"],["F","F","F","F","F",...,"F","F","F","F","F"]]

        """  # NOQA:E501
        # create pyarrow table from the dataframe without the geometry
        if isinstance(geometry_name, str):
            geometry_name = [geometry_name]

        no_null_geoms_df = self._df.filter(
            pl.any_horizontal(pl.col(*geometry_name).is_not_null()),
        )
        if len(no_null_geoms_df) != len(self._df):
            warnings.warn(
                "Dataframe contains null goemetries, nulls will be discarded.",
                stacklevel=2,
            )

        pa_table = self._df.drop(geometry_name).to_arrow()

        for this_g_name in geometry_name:
            crs = pyproj.CRS(self._df[this_g_name].struct.field("crs")[0]).to_wkt(
                version="WKT2_2019",
            )

            # create geoarrow array with crs from the geometry
            geometries_wkb = (
                self._df[this_g_name].struct.field("wkb_geometry").to_numpy().copy()
            )
            geoarrow_geom_array = ga.with_crs(ga.as_geoarrow(geometries_wkb), crs)

            # add the geoarrow geometry to the arrow table
            pa_table = pa_table.append_column(this_g_name, geoarrow_geom_array)
        return pa_table

    def join(
        self,
        other: pl.DataFrame,
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
    ) -> pl.DataFrame:
        r"""Join two SpatialFrames based on a spatial predicate.

        Parameters
        ----------
        other
            SpatialFrame to join with.

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
            Name of the geometry columns in both SpatialFrames.

        left_on
            Name of the geometry column in the left SpatialFrame for the spatial join.

        right_on
            Name of the geometry column in the right SpatialFrame for the spatial join.

        suffix
            Suffix to append to columns with a duplicate name.

        maintain_order
            Which DataFrame row order to preserve, if any.
            Do not rely on any observed ordering without explicitly
            setting this parameter, as your code may break in a future release.
            Not specifying any ordering can improve performance
            Supported for inner, left, right and full joins

            * *none*
                No specific ordering is desired. The ordering might differ across
                Polars versions or even between different runs.
            * *left*
                Preserves the order of the left DataFrame.
            * *right*
                Preserves the order of the right DataFrame.
            * *left_right*
                First preserves the order of the left DataFrame, then the right.
            * *right_left*
                First preserves the order of the right DataFrame, then the left.

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

        self_geometries = self._df[left_on].spatial.to_shapely_array()

        other_geometries = other[right_on].spatial.to_shapely_array()

        tree_query_df = pl.DataFrame(
            shapely.STRtree(other_geometries)
            .query(self_geometries, predicate=predicate, distance=distance)
            .T,
            schema={"left_index": pl.Int64, "right_index": pl.Int64},
        )

        if how in ["left", "right", "full", "inner"]:
            joined = (
                self._df.with_row_index("left_index")
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
                self._df.with_row_index("left_index")
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
        other: pl.DataFrame,
        how: Literal["left", "inner"] = "inner",
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
    ) -> pl.DataFrame:
        r"""Join two dataframes based on a spatial distance.

        Parameters
        ----------
        other
            SpatialFrame to join with.

        how
            Join strategy.

            * *inner*
                Returns rows that have matching values in both tables
            * *left*
                Returns all rows from the left table, and the matched rows from the
                right table

        max_distance
            The maximum distance to search around an input feature.

        on
            Name of the geometry columns in both SpatialFrames.

        left_on
            Name of the geometry column in the left SpatialFrame for the spatial join.

        right_on
            Name of the geometry column in the right SpatialFrame for the spatial join.

        suffix
            Suffix to append to columns with a duplicate name.

        maintain_order
            Which DataFrame row order to preserve, if any.
            Do not rely on any observed ordering without explicitly
            setting this parameter, as your code may break in a future release.
            Not specifying any ordering can improve performance
            Supported for inner, left, right and full joins

            * *none*
                No specific ordering is desired. The ordering might differ across
                Polars versions or even between different runs.
            * *left*
                Preserves the order of the left DataFrame.
            * *right*
                Preserves the order of the right DataFrame.
            * *left_right*
                First preserves the order of the left DataFrame, then the right.
            * *right_left*
                First preserves the order of the right DataFrame, then the left.

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

        self_geometries = self._df[left_on].spatial.to_shapely_array()

        other_geometries = other[right_on].spatial.to_shapely_array()

        query_results = shapely.STRtree(self_geometries).query_nearest(
            other_geometries,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
            all_matches=all_matches,
        )

        if return_distance is True:
            tree_query_df = pl.DataFrame(
                query_results[0].T,
                schema={"right_index": pl.Int64, "left_index": pl.Int64},
            ).with_columns(pl.Series("distance", query_results[1]))
        else:
            tree_query_df = pl.DataFrame(
                query_results,
                schema={"right_index": pl.Int64, "left_index": pl.Int64},
            )

        return (
            self._df.with_row_index("left_index")
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

    def centroid_knn_join(
        self,
        other: pl.DataFrame,
        k: int,
        on: str = "geometry",
        left_on: str | None = None,
        right_on: str | None = None,
        suffix: str = "_right",
    ) -> pl.DataFrame:
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

        Notes
        -----
            As the name implies, this KNN join method only takes into account the
            centroids of the geometries in both dataframes, it may not be suitable
            for joining the nearest lines or polygons depending on the distribution of
            the geometries.

            This method relies on scipy.spatial's KDTree to find the neighbors.

        """
        if left_on is None:
            left_on = on
        if right_on is None:
            right_on = on

        self_df = self._df
        self_centroids = shapely.centroid(self_df[left_on].spatial.to_shapely_array())
        other_centroids = shapely.centroid(other[right_on].spatial.to_shapely_array())

        self_coords = shapely.get_coordinates(self_centroids)
        other_coords = shapely.get_coordinates(other_centroids)

        tree = KDTree(other_coords)
        query_result = tree.query(self_coords, k=k)

        return (
            pl.LazyFrame(query_result[1])
            .with_row_index("self_index")
            .unpivot(
                index="self_index",
                value_name="other_index",
            )
            .drop(
                "variable",
            )
            .join(
                self_df.lazy().with_row_index("self_index"),
                how="left",
                on="self_index",
            )
            .join(
                other.lazy().with_row_index("other_index"),
                how="left",
                on="other_index",
                suffix=suffix,
            )
            .collect(engine="streaming")
        )

    def viz(
        self,
        geometry_name: str = "geometry",
        scatterplot_kwargs: ScatterplotLayerKwargs | None = None,
        path_kwargs: PathLayerKwargs | None = None,
        polygon_kwargs: PolygonLayerKwargs | None = None,
        map_kwargs: MapKwargs | None = None,
    ) -> Map:
        r"""Visualizes the dataframe as a layer in a Lonboard [map][lonboard.Map].

        Parameters
        ----------
        geometry_name
            The name of the column in the dataframe that will be use to visualize the features on the Lonboard map.

        scatterplot_kwargs
            a dict of parameters to pass down to all generated ScatterplotLayers.

        path_kwargs
            a dict of parameters to pass down to all generated PathLayers.

        polygon_kwargs
            a dict of parameters to pass down to all generated PolygonLayers.

        map_kwargs
            a dict of parameters to pass down to the generated Map.

        Note
        ----
        Any rows with null geometries will be discarded.

        Examples
        --------
        >>> from spatial_polars import read_spatial
        >>> my_shapefile = r"c:\data\roads.shp"
        >>> df = read_spatial(my_shapefile)
        >>> df.spatial.viz()

        """  # NOQA:E501
        check_lonboard()
        geoarrow_table = self.to_geoarrow(geometry_name)

        return viz(
            geoarrow_table,
            scatterplot_kwargs=scatterplot_kwargs,
            path_kwargs=path_kwargs,
            polygon_kwargs=polygon_kwargs,
            map_kwargs=map_kwargs,
        )

    def _make_color_array(
        self,
        cmap_col: str,
        cmap_type: Literal["categorical", "continuous"] | None,
        cmap: Palette | Colormap | dict | None,
        alpha: float | NDArray[floating] | None,
        *,
        normalize_cmap_col: bool,
    ) -> NDArray | None:
        check_lonboard()
        color = None
        if cmap_col is not None:
            if cmap_type == "continuous":
                if normalize_cmap_col:
                    norm_arr = (
                        self._df.select(c(cmap_col).spatial.min_max())
                        .to_series()
                        .to_numpy()
                    )
                else:
                    norm_arr = self._df.select(c(cmap_col)).to_series().to_numpy()
                color = apply_continuous_cmap(
                    norm_arr,
                    cmap,
                    alpha=alpha,
                )
            elif cmap_type == "categorical":
                cat_arr = self._df.select(c(cmap_col)).to_series().to_arrow()

                if cmap is None:
                    cmap = {}
                    for cat in self._df[cmap_col].unique():
                        cmap[cat] = [
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        ]

                color = apply_categorical_cmap(
                    cat_arr,
                    cmap,
                    alpha=alpha,
                )
        return color

    def to_scatterplotlayer(
        self,
        geometry_name: str = "geometry",
        *,
        filled: bool = True,
        fill_color: list | tuple | None = None,
        fill_cmap_col: str | None = None,
        fill_cmap_type: Literal["categorical", "continuous"] | None = None,
        fill_cmap: Palette | Colormap | dict | None = None,
        fill_alpha: float | NDArray[floating] | None = None,
        fill_normalize_cmap_col: bool = True,
        stroked: bool = True,
        line_color: list | tuple | None = None,
        line_cmap_col: str | None = None,
        line_cmap_type: Literal["categorical", "continuous"] | None = None,
        line_cmap: Palette | Colormap | dict | None = None,
        line_alpha: float | NDArray[floating] | None = None,
        line_normalize_cmap_col: bool = True,
        line_width: float | NDArray[floating] | str | None = 1,
        line_width_min_pixels: float = 1,
        line_width_max_pixels: float | None = None,
        line_width_scale: float = 1,
        line_width_units: Literal["meters", "common", "pixels"] = "meters",
        radius: float | NDArray[floating] | str | None = 1,
        radius_max_pixels: float | None = None,
        radius_min_pixels: float = 0,
        radius_scale: float = 1,
        radius_units: Literal["meters", "common", "pixels"] = "meters",
        auto_highlight: bool = False,
        highlight_color: list | tuple = (0, 0, 128, 128),
        opacity: float = 1,
        pickable: bool = True,
        visible: bool = True,
        antialiasing: bool = True,
        billboard: bool = False,
        **kwargs: dict[str, Any],
    ) -> ScatterplotLayer:
        """Make a Lonboard [ScatterplotLayer][lonboard.ScatterplotLayer] from the SpatialFrame.

        Parameters
        ----------
        geometry_name
            The name of the column in the SpatialFrame that will be used for the geometries of the points in the layer.

        filled
            Draw the filled area of points.

        fill_color
            The filled color of each object in the format of [r, g, b, [a]]. Each channel is a number between 0-255 and a is 255 if not supplied.

        fill_cmap_col
            The name of the column in the SpatialFrame that will be used to vary the color of the points in the layer.  Only applicable if `fill_cmap_type` is not None.

        fill_cmap_type
            The type of color map to use.  Only applicable if `fill_cmap_col` is set.

        fill_cmap
            If `fill_cmap_type` is `continuous`, The palettable.colorbrewer.diverging colormap used to vary the color of the points in the layer.
            If `fill_cmap_type` is `categorical`, a dictionary of mappings of the values from `fill_cmap_col` to a list of of [r, g, b] color codes, or None. If None, random colors will be selected for each value in `fill_cmap_col`.

        fill_alpha
            The value which will be provided to the alpha chanel of the color for color map.  Only applicable if `fill_cmap_col` and `fill_cmap` are set.

        fill_normalize_cmap_col
            If `True` a copy of the values in fill_cmap_col will be normalized to be between 0-1 for use by Lonboard's `apply_continuous_cmap` function to set the colors of the points in the layer.  If `False`, the values in the column are assumed to already be between 0-1 and do not need to be normalized. Only applicable if `fill_cmap_col` and `fill_cmap` are set and `fill_cmap_type` is `continuous`.

        stroked
            The filled color of each object in the format of

        line_color
            The outline color of each object in the format of [r, g, b, [a]]. Each channel is a number between 0-255 and a is 255 if not supplied.

        line_cmap_col
            The name of the column in the SpatialFrame that will be used to vary the color of the point outlines in the layer.  Only applicable if `line_cmap_type` is not None.

        line_cmap_type
            The type of color map to use.  Only applicable if `line_cmap_col` is set.

        line_cmap
            If `line_cmap_type` is `continuous`, The palettable.colorbrewer.diverging colormap used to vary the color of the point outlines in the layer.
            If `line_cmap_type` is `categorical`, a dictionary of mappings of the values from `line_cmap_col` to a list of of [r, g, b] color codes, or None. If None, random colors will be selected for each value in `line_cmap_col`.

        line_alpha
            The value which will be provided to the alpha chanel of the color for color map.  Only applicable if `line_cmap_col` and `line_cmap` are set.

        line_normalize_cmap_col
            If `True` a copy of the values in line_cmap_col will be normalized to be between 0-1 for use by Lonboard's `apply_continuous_cmap` function to set the colors of the point outlines in the layer.  If `False`, the values in the column are assumed to already be between 0-1 and do not need to be normalized. Only applicable if `line_cmap_col` and `line_cmap` are set and `line_cmap_type` is `continuous`.

        line_width
            The width of each path, in units specified by `width_units` (default 'meters'). If a string is provided, the values from the SpatialFrame in the column with the name will be used.  If a number is provided, it is used as the width for all paths. If an array is provided, each value in the array will be used as the width for the path at the same row index.

        line_width_min_pixels
            The minimum path width in pixels. This prop can be used to prevent the path from getting too thin when zoomed out.

        line_width_max_pixels
            The maximum path width in pixels. This prop can be used to prevent the path from getting too thick when zoomed in.

        line_width_scale
            The path width multiplier that multiplied to all paths.

        line_width_units
            The units of the line width, one of 'meters', 'common', and 'pixels'. See unit system.

        radius
            The radius of each object, in units specified by radius_units (default 'meters').  If a string is provided, the values from the SpatialFrame in the column with the name will be used.  If a number is provided, it is used as the width for all points. If an array is provided, each value in the array will be used as the width for the path at the same row index.

        radius_max_pixels
            The maximum radius in pixels. This can be used to prevent the circle from getting too big when zoomed in.

        radius_min_pixels
            The minimum radius in pixels. This can be used to prevent the circle from getting too small when zoomed out.

        radius_scale
            A global radius multiplier for all points.

        radius_units
            The units of the radius, one of 'meters', 'common', and 'pixels'

        auto_highlight
            When `True`, the current object pointed to by the mouse pointer (when hovered over) is highlighted with highlightColor.  Requires `pickable` to be `True`.

        highlight_color
            RGBA color to blend with the highlighted object (the hovered over object if `auto_highlight`=`True`). When the value is a 3 component (RGB) array, a default alpha of 255 is applied.

        opacity
            The opacity of the layer.

        pickable
            Whether the layer responds to mouse pointer picking events.
            This must be set to `True` for tooltips and other interactive elements to be available. This can also be used to only allow picking on specific layers within a map instance.
            Note that picking has some performance overhead in rendering. To get the absolute best rendering performance with large data (at the cost of removing interactivity), set this to `False`.

        visible
            Whether the layer is visible.
            Under most circumstances, using the `visible` attribute to control the visibility of layers is recommended over removing/adding the layer from the `Map.layers` list.
            In particular, toggling the `visible` attribute will persist the layer on the JavaScript side, while removing/adding the layer from the `Map.layers` list will re-download and re-render from scratch.

        antialiasing
            If True, circles are rendered with smoothed edges. If False, circles are rendered with rough edges. Antialiasing can cause artifacts on edges of overlapping circles.

        billboard
            If True, rendered circles always face the camera. If False circles face up (i.e. are parallel with the ground plane).

        kwargs
            additional kwargs to be supplied to the layer creation such as [layer extensions][lonboard.layer-extensions]

        Note
        ----
        Implementation varies slightly from Lonboard for the setting of color and width to make it easy to use from the SpatialFrame.

        """  # NOQA:E501
        check_lonboard()
        validate_cmap_input(
            self._df,
            fill_cmap_col,
            fill_cmap_type,
            fill_cmap,
        )
        validate_cmap_input(
            self._df,
            line_cmap_col,
            line_cmap_type,
            line_cmap,
        )
        validate_width_and_radius_input(self._df, line_width)
        validate_width_and_radius_input(self._df, radius)
        fill_color = self._make_color_array(
            fill_cmap_col,
            fill_cmap_type,
            fill_cmap,
            fill_alpha,
            normalize_cmap_col=fill_normalize_cmap_col,
        )
        line_color = self._make_color_array(
            line_cmap_col,
            line_cmap_type,
            line_cmap,
            line_alpha,
            normalize_cmap_col=line_normalize_cmap_col,
        )

        if isinstance(line_width, str):
            line_width = self._df.select(c(line_width)).to_series().to_numpy()

        if isinstance(radius, str):
            radius = self._df.select(c(radius)).to_series().to_numpy()

        geoarrow_table = self.to_geoarrow(geometry_name)

        return ScatterplotLayer(
            table=geoarrow_table,
            antialiasing=antialiasing,
            auto_highlight=auto_highlight,
            billboard=billboard,
            filled=filled,
            get_fill_color=fill_color,
            get_line_color=line_color,
            get_line_width=line_width,
            get_radius=radius,
            highlight_color=highlight_color,
            line_width_max_pixels=line_width_max_pixels,
            line_width_min_pixels=line_width_min_pixels,
            line_width_scale=line_width_scale,
            line_width_units=line_width_units,
            opacity=opacity,
            pickable=pickable,
            radius_max_pixels=radius_max_pixels,
            radius_min_pixels=radius_min_pixels,
            radius_scale=radius_scale,
            radius_units=radius_units,
            stroked=stroked,
            visible=visible,
            **kwargs,
        )

    def to_pathlayer(
        self,
        geometry_name: str = "geometry",
        *,
        color: list | tuple | None = None,
        cmap_col: str | None = None,
        cmap_type: Literal["categorical", "continuous"] | None = None,
        cmap: Palette | Colormap | dict | None = None,
        alpha: float | NDArray[floating] | None = None,
        normalize_cmap_col: bool = True,
        width: float | NDArray[floating] | str | None = 1,
        auto_highlight: bool = False,
        billboard: bool = False,
        cap_rounded: bool = False,
        highlight_color: list | tuple = (0, 0, 128, 128),
        joint_rounded: bool = False,
        miter_limit: float = 4,
        opacity: float = 1,
        pickable: bool = True,
        visible: bool = True,
        width_min_pixels: float = 1,
        width_max_pixels: float | None = None,
        width_scale: float = 1,
        width_units: Literal["meters", "common", "pixels"] = "meters",
        **kwargs: dict[str, Any],
    ) -> PathLayer:
        """Make a Lonboard [PathLayer][lonboard.PathLayer] from the SpatialFrame.

        Parameters
        ----------
        geometry_name
            The name of the column in the SpatialFrame that will be used for the geometries of the paths in the layer.

        color
            The color for every path in the format of [r, g, b, [a]]. Each channel is a number between 0-255 and a is 255 if not supplied.

        cmap_col
            The name of the column in the SpatialFrame that will be used to vary the color of the paths in the layer.  Only applicable if `cmap_type` is not None.

        cmap_type
            The type of color map to use.  Only applicable if `cmap_col` is set.

        cmap
            If `cmap_type` is `continuous`, The palettable.colorbrewer.diverging colormap used to vary the color of the lines in the layer.
            If `cmap_type` is `categorical`, a dictionary of mappings of the values from `cmap_col` to a list of of [r, g, b] color codes, or None. If None, random colors will be selected for each value in `cmap_col`.

        alpha
            The value which will be provided to the alpha chanel of the color for color map.  Only applicable if `c_map_col` and `cmap` are set.

        normalize_cmap_col
            If `True` a copy of the values in cmap_col will be normalized to be between 0-1 for use by Lonboard's `apply_continuous_cmap` function to set the colors of the lines in the layer.  If `False`, the values in the column are assumed to already be between 0-1 and do not need to be normalized. Only applicable if `c_map_col` and `cmap` are set and `cmap_type` is `continuous`.

        width
            The width of each path, in units specified by `width_units` (default 'meters'). If a string is provided, the values from the SpatialFrame in the column with the name will be used.  If a number is provided, it is used as the width for all paths. If an array is provided, each value in the array will be used as the width for the path at the same row index.

        pickable
            Whether the layer responds to mouse pointer picking events.
            This must be set to `True` for tooltips and other interactive elements to be available. This can also be used to only allow picking on specific layers within a map instance.
            Note that picking has some performance overhead in rendering. To get the absolute best rendering performance with large data (at the cost of removing interactivity), set this to `False`.

        auto_highlight
            When `True`, the current object pointed to by the mouse pointer (when hovered over) is highlighted with highlightColor.  Requires `pickable` to be `True`.

        highlight_color
            RGBA color to blend with the highlighted object (the hovered over object if `auto_highlight`=`True`). When the value is a 3 component (RGB) array, a default alpha of 255 is applied.

        billboard
            If `True`, extrude the path in screen space (width always faces the camera). If `False`, the width always faces up.

        cap_rounded
            Type of caps. If `True`, draw round caps. Otherwise draw square caps.

        joint_rounded
            Type of joint. If `True`, draw round joints. Otherwise draw miter joints.

        miter_limit
            The maximum extent of a joint in ratio to the stroke width. Only works if jointRounded is `False`.

        opacity
            The opacity of the layer.

        visible
            Whether the layer is visible.
            Under most circumstances, using the `visible` attribute to control the visibility of layers is recommended over removing/adding the layer from the `Map.layers` list.
            In particular, toggling the `visible` attribute will persist the layer on the JavaScript side, while removing/adding the layer from the `Map.layers` list will re-download and re-render from scratch.

        width_min_pixels
            The minimum path width in pixels. This prop can be used to prevent the path from getting too thin when zoomed out.

        width_max_pixels
            The maximum path width in pixels. This prop can be used to prevent the path from getting too thick when zoomed in.

        width_scale
            The path width multiplier that multiplied to all paths.

        width_units
            The units of the line width, one of 'meters', 'common', and 'pixels'. See unit system.

        kwargs
            additional kwargs to be supplied to the layer creation such as [layer extensions][lonboard.layer-extensions]

        Note
        ----
        Implementation varies slightly from Lonboard for the setting of color and width to make it easy to use from the SpatialFrame.

        """  # NOQA:E501
        check_lonboard()
        validate_cmap_input(
            self._df,
            cmap_col,
            cmap_type,
            cmap,
        )
        validate_width_and_radius_input(self._df, width)
        color = self._make_color_array(
            cmap_col,
            cmap_type,
            cmap,
            alpha,
            normalize_cmap_col=normalize_cmap_col,
        )

        if isinstance(width, str):
            width = self._df.select(c(width)).to_series().to_numpy()

        geoarrow_table = self.to_geoarrow(geometry_name)

        return PathLayer(
            table=geoarrow_table,
            auto_highlight=auto_highlight,
            billboard=billboard,
            cap_rounded=cap_rounded,
            get_width=width,
            highlight_color=highlight_color,
            joint_rounded=joint_rounded,
            miter_limit=miter_limit,
            opacity=opacity,
            pickable=pickable,
            visible=visible,
            get_color=color,
            width_min_pixels=width_min_pixels,
            width_max_pixels=width_max_pixels,
            width_scale=width_scale,
            width_units=width_units,
            **kwargs,
        )

    def to_polygonlayer(
        self,
        geometry_name: str = "geometry",
        *,
        filled: bool = True,
        fill_color: list | tuple | None = None,
        fill_cmap_col: str | None = None,
        fill_cmap_type: Literal["categorical", "continuous"] | None = None,
        fill_cmap: Palette | Colormap | dict | None = None,
        fill_alpha: float | NDArray[floating] | None = None,
        fill_normalize_cmap_col: bool = True,
        stroked: bool = True,
        line_color: list | tuple | None = None,
        line_cmap_col: str | None = None,
        line_cmap_type: Literal["categorical", "continuous"] | None = None,
        line_cmap: Palette | Colormap | dict | None = None,
        line_alpha: float | NDArray[floating] | None = None,
        line_normalize_cmap_col: bool = True,
        line_width: float | NDArray[floating] | str | None = 1,
        line_joint_rounded: bool = False,
        line_miter_limit: float = 4,
        line_width_min_pixels: float = 1,
        line_width_max_pixels: float | None = None,
        line_width_scale: float = 1,
        line_width_units: Literal["meters", "common", "pixels"] = "meters",
        elevation: float | NDArray[floating] | str | None = None,
        elevation_scale: float = 1,
        auto_highlight: bool = False,
        highlight_color: list | tuple = (0, 0, 128, 128),
        opacity: float = 1,
        pickable: bool = True,
        visible: bool = True,
        wireframe: bool = False,
        **kwargs: dict[str, Any],
    ) -> PolygonLayer:
        """Make a Lonboard [PolygonLayer][lonboard.PolygonLayer] from the SpatialFrame.

        Parameters
        ----------
        geometry_name
            The name of the column in the SpatialFrame that will be used for the geometries of the polygons in the layer.

        filled
            Whether to draw a filled polygon (solid fill).  Note that only the area between the outer polygon and any holes will be filled.

        fill_color
            The fill color for every polygon in the format of [r, g, b, [a]]. Each channel is a number between 0-255 and a is 255 if not supplied.

        fill_cmap_col
            The name of the column in the SpatialFrame that will be used to vary the color of the polygons in the layer.  Only applicable if `fill_cmap_type` is not None.

        fill_cmap_type
            The type of color map to use.  Only applicable if `fill_cmap_col` is set.

        fill_cmap
            If `fill_cmap_type` is `continuous`, The palettable.colorbrewer.diverging colormap used to vary the color of the polygons in the layer.
            If `fill_cmap_type` is `categorical`, a dictionary of mappings of the values from `fill_cmap_col` to a list of of [r, g, b] color codes, or None. If None, random colors will be selected for each value in `fill_cmap_col`.

        fill_alpha
            The value which will be provided to the alpha chanel of the color for color map.  Only applicable if `fill_cmap_col` and `fill_cmap` are set.

        fill_normalize_cmap_col
            If `True` a copy of the values in fill_cmap_col will be normalized to be between 0-1 for use by Lonboard's `apply_continuous_cmap` function to set the colors of the polygons in the layer.  If `False`, the values in the column are assumed to already be between 0-1 and do not need to be normalized. Only applicable if `fill_cmap_col` and `fill_cmap` are set and `fill_cmap_type` is `continuous`.

        stroked
            Whether to draw an outline around the polygon (solid fill).  Note that both the outer polygon as well the outlines of any holes will be drawn.

        line_color
            The color for every polygon outline in the format of [r, g, b, [a]]. Each channel is a number between 0-255 and a is 255 if not supplied.

        line_cmap_col
            The name of the column in the SpatialFrame that will be used to vary the color of the polygon outlines in the layer.  Only applicable if `line_cmap_type` is not None.

        line_cmap_type
            The type of color map to use.  Only applicable if `line_cmap_col` is set.

        line_cmap
            If `line_cmap_type` is `continuous`, The palettable.colorbrewer.diverging colormap used to vary the color of the polygon outlines in the layer.
            If `line_cmap_type` is `categorical`, a dictionary of mappings of the values from `line_cmap_col` to a list of of [r, g, b] color codes, or None. If None, random colors will be selected for each value in `line_cmap_col`.

        line_alpha
            The value which will be provided to the alpha chanel of the color for color map.  Only applicable if `line_cmap_col` and `line_cmap` are set.

        line_normalize_cmap_col
            If `True` a copy of the values in line_cmap_col will be normalized to be between 0-1 for use by Lonboard's `apply_continuous_cmap` function to set the colors of the polygon outlines in the layer.  If `False`, the values in the column are assumed to already be between 0-1 and do not need to be normalized. Only applicable if `line_cmap_col` and `line_cmap` are set and `line_cmap_type` is `continuous`.

        line_width
            The width of each path, in units specified by `width_units` (default 'meters'). If a string is provided, the values from the SpatialFrame in the column with the name will be used.  If a number is provided, it is used as the width for all paths. If an array is provided, each value in the array will be used as the width for the path at the same row index.

        line_joint_rounded
            Type of joint. If `True`, draw round joints. Otherwise draw miter joints.

        line_miter_limit
            The maximum extent of a joint in ratio to the stroke width. Only works if jointRounded is `False`.

        line_width_min_pixels
            The minimum path width in pixels. This prop can be used to prevent the path from getting too thin when zoomed out.

        line_width_max_pixels
            The maximum path width in pixels. This prop can be used to prevent the path from getting too thick when zoomed in.

        line_width_scale
            The path width multiplier that multiplied to all paths.

        line_width_units
            The units of the line width, one of 'meters', 'common', and 'pixels'. See unit system.

        elevation
            The elevation to extrude each polygon with, in meters.  Only applies if extruded=True.  If a number is provided, it is used as the width for all polygons.  If an array is provided, each value in the array will be used as the width for the polygon at the same row index.  If a string is provided it will be used as a column name in the frame to use for the elevation.
            Providing a value to elevation will set `extruded=True` on the layer.

        elevation_scale
            Elevation multiplier. The final elevation is calculated by elevation_scale * elevation(d). `elevation_scale` is a handy property to scale all elevation without updating the data.

        auto_highlight
            When `True`, the current object pointed to by the mouse pointer (when hovered over) is highlighted with highlightColor.  Requires `pickable` to be `True`.

        highlight_color
            RGBA color to blend with the highlighted object (the hovered over object if `auto_highlight`=`True`). When the value is a 3 component (RGB) array, a default alpha of 255 is applied.

        opacity
            The opacity of the layer.

        pickable
            Whether the layer responds to mouse pointer picking events.
            This must be set to `True` for tooltips and other interactive elements to be available. This can also be used to only allow picking on specific layers within a map instance.
            Note that picking has some performance overhead in rendering. To get the absolute best rendering performance with large data (at the cost of removing interactivity), set this to `False`.

        visible
            Whether the layer is visible.
            Under most circumstances, using the `visible` attribute to control the visibility of layers is recommended over removing/adding the layer from the `Map.layers` list.
            In particular, toggling the `visible` attribute will persist the layer on the JavaScript side, while removing/adding the layer from the `Map.layers` list will re-download and re-render from scratch.

        wireframe
            Whether to generate a line wireframe of the polygon. The outline will have "horizontal" lines closing the top and bottom polygons and a vertical line (a "strut") for each vertex on the polygon.

        kwargs
            additional kwargs to be supplied to the layer creation such as [layer extensions][lonboard.layer-extensions]

        Note
        ----
        Implementation varies slightly from Lonboard for the setting of color and width to make it easy to use from the SpatialFrame.

        """  # NOQA:E501
        check_lonboard()
        validate_cmap_input(
            self._df,
            fill_cmap_col,
            fill_cmap_type,
            fill_cmap,
        )
        validate_cmap_input(
            self._df,
            line_cmap_col,
            line_cmap_type,
            line_cmap,
        )
        validate_width_and_radius_input(self._df, line_width)

        fill_color = self._make_color_array(
            fill_cmap_col,
            fill_cmap_type,
            fill_cmap,
            fill_alpha,
            normalize_cmap_col=fill_normalize_cmap_col,
        )
        line_color = self._make_color_array(
            line_cmap_col,
            line_cmap_type,
            line_cmap,
            line_alpha,
            normalize_cmap_col=line_normalize_cmap_col,
        )

        if isinstance(line_width, str):
            line_width = self._df.select(c(line_width)).to_series().to_numpy()

        extruded = False
        if elevation is not None:
            extruded = True
        if isinstance(elevation, str):
            elevation = self._df.select(c(elevation)).to_series().to_numpy()

        geoarrow_table = self.to_geoarrow(geometry_name)

        return PolygonLayer(
            table=geoarrow_table,
            auto_highlight=auto_highlight,
            elevation_scale=elevation_scale,
            extruded=extruded,
            filled=filled,
            get_elevation=elevation,
            get_fill_color=fill_color,
            get_line_color=line_color,
            get_line_width=line_width,
            highlight_color=highlight_color,
            line_joint_rounded=line_joint_rounded,
            line_miter_limit=line_miter_limit,
            line_width_max_pixels=line_width_max_pixels,
            line_width_min_pixels=line_width_min_pixels,
            line_width_scale=line_width_scale,
            line_width_units=line_width_units,
            opacity=opacity,
            pickable=pickable,
            stroked=stroked,
            visible=visible,
            wireframe=wireframe,
            **kwargs,
        )

    @staticmethod
    def from_point_coords(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str | None = None,
        crs: Any = 4326,  # NOQA:ANN401
    ) -> pl.DataFrame:
        r"""Create a SpatialFrame from a polars DataFrame with x/y/(z) columns.

        Parameters
        ----------
        df
            The dataframe which contains the data from which to create the SpatialFrame.

        x_col
            The name of the column in the DataFrame which holds the X coordinates.

        y_col
            The name of the column in the DataFrame which holds the Y coordinates.

        z_col
            The name of the column in the DataFrame which holds the Z coordinates.

        crs
            A crs representation that can be provided to pyproj.CRS.from_user_input to produce a CRS.

            PROJ string

            Dictionary of PROJ parameters

            PROJ keyword arguments for parameters

            JSON string with PROJ parameters

            CRS WKT string

            An authority string [i.e. `epsg:4326`]

            An EPSG integer code [i.e. 4326]

            A tuple of (“auth_name”: “auth_code”) [i.e (`epsg`, `4326`)]

            An object with a to_wkt method.

            A pyproj.crs.CRS class

        Examples
        --------
        Creating a SpatialFrame from a polars df with a columns of coordinates of points .

        >>> import polars as pl
        >>> from spatial_polars import SpatialFrame
        >>> df = pl.DataFrame({
        >>>     "Place":["Gateway Arch", "Monks Mound"],
        >>>     "x":[-90.18497, -90.06211],
        >>>     "y":[38.62456, 38.66072],
        >>>     "z":[0,0]
        >>> })
        >>> s_df = SpatialFrame.from_point_coords(df, "x", "y", "z")
        >>> s_df
        shape: (2, 2)
        ┌──────────────┬─────────────────────────────────┐
        │ Place        ┆ geometry                        │
        │ ---          ┆ ---                             │
        │ str          ┆ struct[2]                       │
        ╞══════════════╪═════════════════════════════════╡
        │ Gateway Arch ┆ {b"\x01\x01\x00\x00\x80o/i\x8c… │
        │ Monks Mound  ┆ {b"\x01\x01\x00\x00\x80K\xb08\… │
        └──────────────┴─────────────────────────────────┘

        """  # NOQA:E501
        coord_cols = [x_col, y_col]
        if z_col is not None:
            coord_cols.append(z_col)

        points = shapely.points(df.select(coord_cols).to_numpy().copy())
        wkb_array = shapely.to_wkb(points)
        crs_wkt = pyproj.CRS.from_user_input(crs).to_wkt()
        return df.drop(coord_cols).with_columns(
            pl.struct(
                pl.Series("wkb_geometry", wkb_array, dtype=pl.Binary),
                pl.lit(crs_wkt, dtype=pl.Categorical).alias("crs"),
            ).alias("geometry"),
        )

    @staticmethod
    def from_WKB(df: pl.DataFrame, wkb_col: str, crs: Any = 4326) -> pl.DataFrame:  # NOQA:ANN401 N802
        r"""Create a SpatialFrame from a polars DataFrame with a column of WKB.

        Parameters
        ----------
        df
            The dataframe which contains the data from which to create the SpatialFrame.

        wkb_col
            The name of the column in the DataFrame which holds geometry WKB.

        crs
            A crs representation that can be provided to pyproj.CRS.from_user_input to produce a CRS.

            PROJ string

            Dictionary of PROJ parameters

            PROJ keyword arguments for parameters

            JSON string with PROJ parameters

            CRS WKT string

            An authority string [i.e. 'epsg:4326']

            An EPSG integer code [i.e. 4326]

            A tuple of (“auth_name”: “auth_code”) [i.e ('epsg', '4326')]

            An object with a to_wkt method.

            A pyproj.crs.CRS class

        Examples
        --------
        Creating a SpatialFrame from a polars df with a column of WKB.

        >>> import polars as pl
        >>> import shapely
        >>> from spatial_polars import SpatialFrame
        >>> arch_wkb = shapely.Point(-90.18497, 38.62456).wkb
        >>> monks_mound_wkb = shapely.Point(-90.06211, 38.66072).wkb
        >>> df = pl.DataFrame({
        >>>     "Place":["Gateway Arch", "Monks Mound"],
        >>>     "wkb":[arch_wkb, monks_mound_wkb],
        >>> })
        >>> s_df = SpatialFrame.from_WKB(df, "wkb")
        >>> s_df
        shape: (2, 2)
        ┌──────────────┬─────────────────────────────────┐
        │ Place        ┆ geometry                        │
        │ ---          ┆ ---                             │
        │ str          ┆ struct[2]                       │
        ╞══════════════╪═════════════════════════════════╡
        │ Gateway Arch ┆ {b"\x01\x01\x00\x00\x80o/i\x8c… │
        │ Monks Mound  ┆ {b"\x01\x01\x00\x00\x80K\xb08\… │
        └──────────────┴─────────────────────────────────┘


        """  # NOQA:E501
        return df.with_columns(
            c(wkb_col).spatial.from_WKB(crs),
        )

    @staticmethod
    def from_WKT(df: pl.DataFrame, wkt_col: str, crs: Any = 4326) -> pl.DataFrame:  # NOQA:ANN401 N802
        r"""Create a SpatialFrame from a polars DataFrame with a column of WKT.

        Parameters
        ----------
        df
            The dataframe which contains the data from which to create the SpatialFrame.

        wkt_col
            The name of the column in the DataFrame which holds geometry WKT.

        crs
            A crs representation that can be provided to pyproj.CRS.from_user_input to produce a CRS.

            PROJ string

            Dictionary of PROJ parameters

            PROJ keyword arguments for parameters

            JSON string with PROJ parameters

            CRS WKT string

            An authority string [i.e. `epsg:4326`]

            An EPSG integer code [i.e. 4326]

            A tuple of (“auth_name”: “auth_code”) [i.e (`epsg`, `4326`)]

            An object with a to_wkt method.

            A pyproj.crs.CRS class

        Examples
        --------
        Creating a SpatialFrame from a polars df with a column of WKT.

        >>> import polars as pl
        >>> import shapely
        >>> from spatial_polars import SpatialFrame
        >>> arch_wkt = shapely.Point(-90.18497, 38.62456).wkt
        >>> monks_mound_wkt = shapely.Point(-90.06211, 38.66072).wkt
        >>> df = pl.DataFrame({
        >>>     "Place":["Gateway Arch", "Monks Mound"],
        >>>     "wkt":[arch_wkt, monks_mound_wkt],
        >>> })
        >>> s_df = SpatialFrame.from_WKT(df, "wkt")
        >>> s_df
        shape: (2, 2)
        ┌──────────────┬─────────────────────────────────┐
        │ Place        ┆ geometry                        │
        │ ---          ┆ ---                             │
        │ str          ┆ struct[2]                       │
        ╞══════════════╪═════════════════════════════════╡
        │ Gateway Arch ┆ {b"\x01\x01\x00\x00\x80o/i\x8c… │
        │ Monks Mound  ┆ {b"\x01\x01\x00\x00\x80K\xb08\… │
        └──────────────┴─────────────────────────────────┘

        """  # NOQA:E501
        return df.with_columns(
            c(wkt_col).spatial.from_WKT(crs),
        )
