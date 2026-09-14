"""io.

This module provides functions for creating polars dataframes from spatial sources.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
)

import polars as pl
import pyarrow.parquet as _pq
import pyogrio
import pyproj
from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    from collections.abc import Iterator
    from io import BytesIO
    from typing import TypeAlias

    import shapely

__all__ = [
    "read_spatial",
    "scan_spatial",
    "spatial_series_dtype",
]

FileSource: TypeAlias = (
    str
    | Path
    | IO[bytes]
    | bytes
    | list[str]
    | list[Path]
    | list[IO[bytes]]
    | list[bytes]
)

pl.register_extension_type("geoarrow.wkb", as_storage=True)
spatial_series_dtype = pl.Struct({"wkb_geometry": pl.Binary, "crs": pl.Categorical})

PYOGRIO_POLARS_DTYPES = {
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int": pl.Int64,
    "int64": pl.Int64,
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint": pl.UInt64,
    "uint64": pl.UInt64,
    "bool": pl.Boolean,
    "float32": pl.Float32,
    "float": pl.Float64,
    "float64": pl.Float64,
    "datetime64[D]": pl.Date,
    "datetime64[us]": pl.Datetime("us"),
    "datetime64[ns]": pl.Datetime("ns"),
    "datetime64[ms]": pl.Datetime("ms"),
    "datetime64": pl.Datetime("ms"),  # ms??
    "object": pl.String,
}


def scan_geoparquet(source: FileSource, kwargs: Any=None) -> pl.LazyFrame:  # noqa: ANN401
    """Scan a geoparquet file and return a spatial LazyFrame.

    A wrapper around pl.scan_parquet that parses the geoparquet metadata and adds the
    crs of all geometry columns to the geometry struct.

    Parameters
    ----------
    source
        Path(s) to a file or directory When needing to authenticate for scanning cloud
        locations, see the storage_options parameter.

    kwargs
        Additional keyword arguments to be passed to [pl.scan_parquet](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)

    """
    if kwargs is None:
        kwargs = {}
    lf = pl.scan_parquet(source, **kwargs)

    table = _pq.read_table(source)
    table_metadata = table.schema.metadata
    if table_metadata is None:
        warnings.warn(
            "No schema metadata found in parquet file. No columns"
            " converted to spatial polars geometry struct.",
            stacklevel=2,
        )
        return lf
    if b"geo" not in table_metadata:
        warnings.warn(
            "No geoparquet metadata found in parquet file. No columns"
            " converted to spatial polars geometry struct.",
            stacklevel=2,
        )
        return lf

    geo_meta = json.loads(table_metadata[b"geo"])
    columns_meta = geo_meta["columns"]

    for geom_col_name, geom_col_meta in columns_meta.items():
        if "crs" in geom_col_meta:
            # convert whatever it is WKT2_2019
            crs_wkt = pyproj.CRS.from_user_input(geom_col_meta["crs"]).to_wkt(
                "WKT2_2019",
            )
        else:
            # crs is optional, default is OGC:CRS84
            crs_wkt = pyproj.CRS.from_user_input("OGC:CRS84").to_wkt(
                "WKT2_2019",
            )
        lf = lf.with_columns(
            pl.col(geom_col_name).spatial.from_WKB(crs_wkt),
        )
    return lf

def read_geoparquet(source: FileSource, kwargs: Any=None)-> pl.DataFrame:  # noqa: ANN401
    """Read a geoparquet file and return a spatial DataFrame.

    A wrapper around `scan_geoparquet` that immediately collects the dataframe.

    Parameters
    ----------
    source
        Path(s) to a file or directory When needing to authenticate for scanning cloud
        locations, see the storage_options parameter.

    kwargs
        Additional keyword arguments to be passed to [pl.scan_parquet](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)

    """
    if kwargs is None:
        kwargs = {}
    return scan_geoparquet(source, kwargs).collect()

def scan_spatial(  # NOQA:C901
    path_or_buffer: str | Path | BytesIO,
    layer: str | int | None = None,
    encoding: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    mask: shapely.Polygon | None = None,
) -> pl.LazyFrame:
    r"""Scan a data source [supported by pyogrio](https://pyogrio.readthedocs.io/en/stable/supported_formats.html) to produce a polars LazyFrame.

    Note
    ----
    To scan a geoparquet file, use the `scan_geoparquet` function.

    Parameters
    ----------
    path_or_buffer
        A dataset path or URI, raw buffer, or file-like object with a read method.

    layer
        If an integer is provided, it corresponds to the index of the layer with the
        data source. If a string is provided, it must match the name of the layer in
        the data source. Defaults to first layer in data source.

    encoding
        If present, will be used as the encoding for reading string values from the
        data source. By default will automatically try to detect the native encoding
        and decode to UTF-8.

    bbox
        If present, will be used to filter records whose geometry intersects this
        box. This must be in the same CRS as the dataset. If GEOS is present and
        used by GDAL, only geometries that intersect this bbox will be returned;
        if GEOS is not available or not used by GDAL, all geometries with bounding
        boxes that intersect this bbox will be returned. Cannot be combined with mask
        keyword.  Tuple should be in the format of (xmin, ymin, xmax, ymax).

    mask
        If present, will be used to filter records whose geometry intersects this
        geometry. This must be in the same CRS as the dataset. If GEOS is present
        and used by GDAL, only geometries that intersect this geometry will be
        returned; if GEOS is not available or not used by GDAL, all geometries with
        bounding boxes that intersect the bounding box of this geometry will be
        returned. Requires Shapely >= 2.0. Cannot be combined with bbox keyword.

    Examples
    --------
    **Scanning a layer from a geopackage:**

    >>> from spatial_polars import scan_spatial
    >>> my_geopackage = r"c:\data\hiking_club.gpkg"
    >>> lf = scan_spatial(my_geopackage, layer="hike")
    >>> lf
    naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)
    PYTHON SCAN []
    PROJECT */4 COLUMNS

    **Scanning a shapefile:**

    >>> from spatial_polars import scan_spatial
    >>> my_shapefile = r"c:\data\roads.shp"
    >>> lf = scan_spatial(my_shapefile)
    >>> lf
    naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)
    PYTHON SCAN []
    PROJECT */11 COLUMNS

    **Scanning a shapefile from within a zipped directory:**

    >>> from spatial_polars import scan_spatial
    >>> zipped_shapefiles = r"C:\data\illinois-latest-free.shp.zip"
    >>> lf = scan_spatial(zipped_shapefiles, layer="gis_osm_roads_free_1")
    >>> lf
    naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)
    PYTHON SCAN []
    PROJECT */11 COLUMNS

    """  # NOQA:E501
    if isinstance(path_or_buffer, (str, Path)) and str(path_or_buffer).endswith(
        ".parquet",
    ):
        warnings.warn(
            "For geoparquet data, use `scan_geoparquet()` or `read_geoparquet()`.",
            stacklevel=2,
        )
    layer_info = pyogrio.read_info(path_or_buffer, layer=layer, encoding=encoding)
    schema = dict(
        zip(
            layer_info["fields"],
            [PYOGRIO_POLARS_DTYPES[dt] for dt in layer_info["dtypes"]],
            strict=True,
        ),
    )

    pyogrio_g_col_name = None
    ret_g_col_name = None
    if layer_info.get("fid_column"):
        schema[layer_info.get("fid_column")] = pl.Int64
    if layer_info.get("geometry_type"):
        pyogrio_g_col_name = layer_info["geometry_name"]
        if pyogrio_g_col_name == "":
            pyogrio_g_col_name = "wkb_geometry"
            ret_g_col_name = "geometry"
        else:
            ret_g_col_name = pyogrio_g_col_name
        schema[ret_g_col_name] = spatial_series_dtype

    def source_generator(  # NOQA:C901,PLR0912
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """Create the source.

        This function will be registered as IO source.
        """
        return_fids = False

        if batch_size is None:
            batch_size = 100

        if with_columns is None:
            read_geometry = True
            return_fids = True
        elif ret_g_col_name in with_columns:
            read_geometry = True
            with_columns.remove(ret_g_col_name)
        else:
            read_geometry = False

        if (
            with_columns is not None
            and layer_info.get("fid_column") in with_columns
        ):
            return_fids = True
            with_columns.remove(layer_info.get("fid_column"))

        with pyogrio.open_arrow(
            path_or_buffer,
            layer=layer,
            encoding=encoding,
            columns=with_columns,
            return_fids=return_fids,
            read_geometry=read_geometry,
            force_2d=False,
            bbox=bbox,
            mask=mask,
            batch_size=batch_size,
            use_pyarrow=True,
        ) as source:
            meta, reader = source

            if read_geometry is True and ret_g_col_name:
                # extract the crs from the metadata
                crs_wkt = pyproj.CRS(meta["crs"]).to_wkt()

            while n_rows is None or n_rows > 0:
                for batch in reader:
                    if n_rows is not None and n_rows <= 0:
                        break

                    batch_df = pl.DataFrame(batch[0:n_rows])

                    if read_geometry and ret_g_col_name:
                        if pyogrio_g_col_name != ret_g_col_name:
                            batch_df = batch_df.with_columns(
                                pl.col(pyogrio_g_col_name).spatial.from_WKB(crs_wkt).alias(ret_g_col_name),
                            ).drop(
                                pl.col(pyogrio_g_col_name),
                            )
                        else:
                            batch_df = batch_df.with_columns(
                                pl.col(pyogrio_g_col_name).spatial.from_WKB(crs_wkt),
                            )

                    if n_rows is not None:
                        n_rows -= batch_df.height

                    if predicate is not None:
                        batch_df = batch_df.filter(predicate)

                    yield batch_df
                if n_rows is None or n_rows <= 0:
                    break
    return register_io_source(io_source=source_generator, schema=schema)


def read_spatial(
    path_or_buffer: str | Path | BytesIO,
    layer: str | int | None = None,
    encoding: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    mask: shapely.Polygon | None = None,
) -> pl.DataFrame:
    r"""Read a spatial data source [supported by pyogrio](https://pyogrio.readthedocs.io/en/stable/supported_formats.html) to produce a polars DataFrame.

    Note
    ----
    Although geoparquet is supported, this implementation, in its current state, leaves
    a lot to be desired.

    Parameters
    ----------
    path_or_buffer
        A dataset path or URI, raw buffer, or file-like object with a read method.

    layer
        If an integer is provided, it corresponds to the index of the layer with the
        data source. If a string is provided, it must match the name of the layer in
        the data source. Defaults to first layer in data source.

    encoding
        If present, will be used as the encoding for reading string values from the
        data source. By default will automatically try to detect the native encoding
        and decode to UTF-8.

    bbox
        If present, will be used to filter records whose geometry intersects this
        box. This must be in the same CRS as the dataset. If GEOS is present and
        used by GDAL, only geometries that intersect this bbox will be returned;
        if GEOS is not available or not used by GDAL, all geometries with bounding
        boxes that intersect this bbox will be returned. Cannot be combined with mask
        keyword.  Tuple should be in the format of (xmin, ymin, xmax, ymax).

    mask
        If present, will be used to filter records whose geometry intersects this
        geometry. This must be in the same CRS as the dataset. If GEOS is present
        and used by GDAL, only geometries that intersect this geometry will be
        returned; if GEOS is not available or not used by GDAL, all geometries with
        bounding boxes that intersect the bounding box of this geometry will be
        returned. Requires Shapely >= 2.0. Cannot be combined with bbox keyword.

    Examples
    --------
    **Scanning a layer from a geopackage:**

    >>> from spatial_polars import read_spatial
    >>> my_geopackage = r"c:\data\hiking_club.gpkg"
    >>> df = read_spatial(my_geopackage, layer="hike")
    >>> df
    shape: (31, 4)
    ┌─────────────────────────────────┬────────────┬──────────┬─────────────────────────────────┐
    │ LOCATION                        ┆ DATE       ┆ DISTANCE ┆ geometry                        │
    │ ---                             ┆ ---        ┆ ---      ┆ ---                             │
    │ str                             ┆ date       ┆ f64      ┆ struct[2]                       │
    ╞═════════════════════════════════╪════════════╪══════════╪═════════════════════════════════╡
    │ Watershed Nature Center         ┆ 2023-01-14 ┆ 1.25     ┆ {b"\x01\x02\x00\x00\x00\xd8\x0… │
    │ Ellis Island                    ┆ 2023-03-11 ┆ 2.25     ┆ {b"\x01\x02\x00\x00\x00\x82\x0… │
    │ Cahokia Mounds State Historic … ┆ 2023-02-04 ┆ 1.75     ┆ {b"\x01\x02\x00\x00\x00\xb1\x0… │
    │ Willoughby Heritage Farm        ┆ 2022-12-03 ┆ 0.75     ┆ {b"\x01\x02\x00\x00\x00\xef\x0… │
    │ Pere Marquette State Park       ┆ 2022-10-15 ┆ 1.0      ┆ {b"\x01\x02\x00\x00\x002\x02\x… │
    │ …                               ┆ …          ┆ …        ┆ …                               │
    │ Haunted Glen Carbon             ┆ 2024-10-19 ┆ 1.75     ┆ {b"\x01\x02\x00\x00\x00\x82\x0… │
    │ Watershed Nature Center         ┆ 2024-10-08 ┆ 1.0      ┆ {b"\x01\x02\x00\x00\x000\x00\x… │
    │ Beaver Dam State Park           ┆ 2024-10-26 ┆ 2.0      ┆ {b"\x01\x02\x00\x00\x00\xc4\x0… │
    │ Willoughby Heritage Farm        ┆ 2024-12-07 ┆ 1.5      ┆ {b"\x01\x02\x00\x00\x00>\x02\x… │
    │ Cahokia Mounds State Historic … ┆ 2025-03-08 ┆ 1.75     ┆ {b"\x01\x02\x00\x00\x00\xeb\x0… │
    └─────────────────────────────────┴────────────┴──────────┴─────────────────────────────────┘

    **Scanning a shapefile:**

    >>> my_shapefile = r"c:\data\roads.shp"
    >>> df = read_spatial(my_shapefile)
    >>> df
    shape: (1_662_837, 11)
    ┌────────────┬──────┬─────────────┬─────────────┬───┬───────┬────────┬────────┬────────────────────┐
    │ osm_id     ┆ code ┆ fclass      ┆ name        ┆ … ┆ layer ┆ bridge ┆ tunnel ┆ geometry           │
    │ ---        ┆ ---  ┆ ---         ┆ ---         ┆   ┆ ---   ┆ ---    ┆ ---    ┆ ---                │
    │ str        ┆ i32  ┆ str         ┆ str         ┆   ┆ i64   ┆ str    ┆ str    ┆ struct[2]          │
    ╞════════════╪══════╪═════════════╪═════════════╪═══╪═══════╪════════╪════════╪════════════════════╡
    │ 4265057    ┆ 5114 ┆ secondary   ┆ 55th Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x03\x0…      │
    │ 4265058    ┆ 5114 ┆ secondary   ┆ Fairview    ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆ Avenue      ┆   ┆       ┆        ┆        ┆ 0\x00\x0e\x0…      │
    │ 4267607    ┆ 5114 ┆ secondary   ┆ 31st Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x02\x0…      │
    │ 4271616    ┆ 5115 ┆ tertiary    ┆ 59th Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x15\x0…      │
    │ 4275365    ┆ 5122 ┆ residential ┆ 61st Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00"\x00\x…      │
    │ …          ┆ …    ┆ …           ┆ …           ┆ … ┆ …     ┆ …      ┆ …      ┆ …                  │
    │ 1370383592 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x02\x0…      │
    │ 1370383593 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x07\x0…      │
    │ 1370383594 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x1c\x0…      │
    │ 1370383595 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x0b\x0…      │
    │ 1370398885 ┆ 5141 ┆ service     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x02\x0…      │
    └────────────┴──────┴─────────────┴─────────────┴───┴───────┴────────┴────────┴────────────────────┘

    **Scanning a shapefile from within a zipped directory:**

    >>> zipped_shapefiles = r"C:\data\illinois-latest-free.shp.zip"
    >>> df = read_spatial(zipped_shapefiles, layer="gis_osm_roads_free_1")
    >>> df
    shape: (1_662_837, 11)
    ┌────────────┬──────┬─────────────┬─────────────┬───┬───────┬────────┬────────┬────────────────────┐
    │ osm_id     ┆ code ┆ fclass      ┆ name        ┆ … ┆ layer ┆ bridge ┆ tunnel ┆ geometry           │
    │ ---        ┆ ---  ┆ ---         ┆ ---         ┆   ┆ ---   ┆ ---    ┆ ---    ┆ ---                │
    │ str        ┆ i32  ┆ str         ┆ str         ┆   ┆ i64   ┆ str    ┆ str    ┆ struct[2]          │
    ╞════════════╪══════╪═════════════╪═════════════╪═══╪═══════╪════════╪════════╪════════════════════╡
    │ 4265057    ┆ 5114 ┆ secondary   ┆ 55th Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x03\x0…      │
    │ 4265058    ┆ 5114 ┆ secondary   ┆ Fairview    ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆ Avenue      ┆   ┆       ┆        ┆        ┆ 0\x00\x0e\x0…      │
    │ 4267607    ┆ 5114 ┆ secondary   ┆ 31st Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x02\x0…      │
    │ 4271616    ┆ 5115 ┆ tertiary    ┆ 59th Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x15\x0…      │
    │ 4275365    ┆ 5122 ┆ residential ┆ 61st Street ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00"\x00\x…      │
    │ …          ┆ …    ┆ …           ┆ …           ┆ … ┆ …     ┆ …      ┆ …      ┆ …                  │
    │ 1370383592 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x02\x0…      │
    │ 1370383593 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x07\x0…      │
    │ 1370383594 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x1c\x0…      │
    │ 1370383595 ┆ 5153 ┆ footway     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x0b\x0…      │
    │ 1370398885 ┆ 5141 ┆ service     ┆ null        ┆ … ┆ 0     ┆ F      ┆ F      ┆ {b"\x01\x02\x00\x0 │
    │            ┆      ┆             ┆             ┆   ┆       ┆        ┆        ┆ 0\x00\x02\x0…      │
    └────────────┴──────┴─────────────┴─────────────┴───┴───────┴────────┴────────┴────────────────────┘

    """  # NOQA:E501
    return scan_spatial(
        path_or_buffer=path_or_buffer,
        layer=layer,
        encoding=encoding,
        bbox=bbox,
        mask=mask,
    ).collect()
