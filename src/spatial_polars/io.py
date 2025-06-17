from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

import polars as pl
import pyogrio
import pyproj
import shapely
import pyarrow.parquet as _pq
import json

from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    from io import BytesIO

__all__ = [
    "spatial_series_dtype",
    "scan_spatial",
    "read_spatial",
]

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


def scan_spatial(
    path_or_buffer: str | Path | BytesIO,
    layer: Optional[str | int] = None,
    encoding: Optional[str] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    mask: Optional[shapely.Polygon] = None,
) -> pl.LazyFrame:
    r"""
    Scans a data source [supported by pyogrio](https://pyogrio.readthedocs.io/en/stable/supported_formats.html) or a geoparquet file to produce a polars LazyFrame.

    Note
    ----
    Although geoparquet is supported, this implementation, in its current state, leaves a lot to be desired.

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

    >>> zipped_shapefiles = r"C:\data\illinois-latest-free.shp.zip"
    >>> lf = scan_spatial(zipped_shapefiles, layer="gis_osm_roads_free_1")
    >>> lf
    naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)
    PYTHON SCAN []
    PROJECT */11 COLUMNS
    """
    if isinstance(path_or_buffer, (str, Path)) and str(path_or_buffer).endswith(
        ".parquet"
    ):
        # TODO look into libgdal-arrow-parquet from conda forge
        # https://pyogrio.readthedocs.io/en/latest/install.html#conda-forge
        schema = pl.scan_parquet(path_or_buffer).collect_schema()
        if schema.get("geometry") is not None:
            schema["geometry"] = spatial_series_dtype
        if bbox is not None:
            mask = shapely.Polygon(shapely.box(*bbox))

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            """
            Generator function that creates the source.
            This function will be registered as IO source.
            """
            if mask is not None:
                if "geometry" not in with_columns:
                    with_columns.append("geometry")

            tab = _pq.read_table(path_or_buffer)
            tab_metadata = tab.schema.metadata if tab.schema.metadata else {}
            if b"geo" in tab_metadata:
                geo_meta = json.loads(tab_metadata[b"geo"])
            else:
                geo_meta = {}
            geom_col = geo_meta["primary_column"]
            crs_wkt = pyproj.CRS(geo_meta["columns"][geom_col]["crs"]).to_wkt()

            if batch_size is None:
                batch_size = 10000

            if with_columns is None:
                read_geometry = True
            elif "geometry" in with_columns:
                read_geometry = True
            else:
                read_geometry = False

            lf = pl.scan_parquet(path_or_buffer)

            if with_columns is not None:
                lf = lf.select(with_columns)

            if predicate is not None:
                lf = lf.filter(predicate)

            previous_max = 0
            while n_rows is None or n_rows > 0:
                batch = lf.slice(previous_max, previous_max + batch_size).collect()
                if batch.height is None or batch.height == 0:
                    break
                if n_rows is not None and n_rows <= 0:
                    break

                if read_geometry:
                    # get the geometries from the batch
                    geometries = batch[0:n_rows][geom_col]
                    shapely_goms = shapely.from_wkb(geometries)
                    geometries = shapely.to_wkb(shapely_goms)

                    # create the dataframe with the non geometry columns
                    # then add struct column with the WKB geometries/CRS
                    df = pl.DataFrame(batch[0:n_rows].drop(geom_col)).with_columns(
                        pl.struct(
                            pl.Series("wkb_geometry", geometries, dtype=pl.Binary),
                            pl.lit(crs_wkt, dtype=pl.Categorical).alias("crs"),
                        ).alias("geometry")
                    )
                else:
                    df = pl.DataFrame(batch[0:n_rows])
                previous_max += df.height

                if n_rows is not None:
                    n_rows -= df.height

                if predicate is not None:
                    df = df.filter(predicate)

                if mask is not None:
                    df = df.filter(pl.col("geometry").spatial.intersects(mask))
                if mask is not None:
                    if "geometry" not in with_columns:
                        df = df.drop("geometry")

                yield df

    else:
        # not geoparquet
        layer_info = pyogrio.read_info(path_or_buffer, layer=layer, encoding=encoding)
        schema = dict(
            zip(
                layer_info["fields"],
                [PYOGRIO_POLARS_DTYPES[dt] for dt in layer_info["dtypes"]],
            )
        )
        if layer_info.get("fid_column"):

            schema[layer_info.get("fid_column")] = pl.Int64
        if layer_info.get("geometry_type"):
            schema["geometry"] = spatial_series_dtype

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            """
            Generator function that creates the source.
            This function will be registered as IO source.
            """
            return_fids = False

            if batch_size is None:
                batch_size = 100

            if with_columns is None:
                read_geometry = True
                return_fids = True
            elif "geometry" in with_columns:
                read_geometry = True
                with_columns.remove("geometry")
            else:
                read_geometry = False
            
            if with_columns is not None and layer_info.get("fid_column") in with_columns:
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
                
                if read_geometry is True and layer_info.get("geometry_type"):
                    # extract the crs from the metadata
                    crs_wkt = pyproj.CRS(meta["crs"]).to_wkt()
                    geom_col = meta["geometry_name"] or "wkb_geometry"

                while n_rows is None or n_rows > 0:
                    for batch in reader:
                        if n_rows is not None and n_rows <= 0:
                            break

                        if read_geometry and layer_info.get("geometry_type"):
                            # get the geometries from the batch
                            geometries = batch[geom_col][0:n_rows]
                            shapely_goms = shapely.from_wkb(geometries)
                            geometries = shapely.to_wkb(shapely_goms)
                            # create the dataframe with the non geometry columns
                            # then add struct column with the WKB geometries/CRS
                            df = pl.DataFrame(
                                batch[0:n_rows].drop_columns(geom_col)
                            ).with_columns(
                                pl.struct(
                                    pl.Series(
                                        "wkb_geometry", geometries, dtype=pl.Binary
                                    ),
                                    pl.lit(crs_wkt, dtype=pl.Categorical).alias("crs"),
                                ).alias("geometry")
                            )
                        else:
                            df = pl.DataFrame(batch[0:n_rows])

                        if n_rows is not None:
                            n_rows -= df.height

                        if predicate is not None:
                            df = df.filter(predicate)

                        yield df
                    if n_rows is None or n_rows <= 0:
                        break

    return register_io_source(io_source=source_generator, schema=schema)


def read_spatial(
    path_or_buffer: str | Path | BytesIO,
    layer: Optional[str | int] = None,
    encoding: Optional[str] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    mask: Optional[shapely.Polygon] = None,
) -> pl.DataFrame:
    r"""
    Reads a spatial data source [supported by pyogrio](https://pyogrio.readthedocs.io/en/stable/supported_formats.html) to produce a polars DataFrame.

    Note
    ----
    Although geoparquet is supported, this implementation, in its current state, leaves a lot to be desired.

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

    """
    return scan_spatial(
        path_or_buffer=path_or_buffer,
        layer=layer,
        encoding=encoding,
        bbox=bbox,
        mask=mask,
    ).collect(engine="streaming")
