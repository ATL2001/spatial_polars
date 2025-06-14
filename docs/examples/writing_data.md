# Writing Data

Spatial polars provides an few options to write dataframes to spatial formats.  Like reading data, spatial polars uses [pyogrio](https://pyogrio.readthedocs.io/en/latest/introduction.html) under the hood when using the `write_spatial` method to write data, so [anything it can write](https://pyogrio.readthedocs.io/en/latest/supported_formats.html) should be fine.

Spatial polars can also write to geoparquet with help from [GeoArrow Python](https://geoarrow.org/geoarrow-python/main/index.html) using the `write_geoparquet` method.

--8<-- "geodatasets_note.md"

Before we can write something, we need something **to** write, so we'll read some data from the geodatasets package into a dataframe.

```py title="setup"
--8<-- "writing_data.py:setup"
```

### Writing to a shapefile

The following example will demonstrate how easy it is to write data to an ESRI Shapefile in your temporary directory.

```py title="Writing to a shapefile"  hl_lines="3"
--8<-- "writing_data.py:shapefile"
```

### Writing to a geopackage

The example will demonstrate how to write data to a geopackage.

You'll notice that this is slightly more complicated than writing to a shapefile because a geopackage can store more than one table, so we need to provide the name for the table to the `layer` parameter.

```py title="Writing to a geopackage"  hl_lines="3"
--8<-- "writing_data.py:geopackage"
```

### Appending to an existing dataset
Additionaly, many formats also support appending new records to an existing dataset.  This is done simply by adding `append=True` to our `write_spatial` call.

```py title="appending to a geopackage"
--8<-- "writing_data.py:append"
```

### Writing to geoparquet

The following example will demonstrate how to write data to a geoparquet file.

```py title="writing to geoparquet" hl_lines="3"
--8<-- "writing_data.py:geoparquet"
```