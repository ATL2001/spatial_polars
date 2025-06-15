# Spatial Polars
A package that extends [polars](https://pola.rs) for working with geospatial data in polars at blazing speed:rocket:.

Spatial polars relies on [polars](https://pola.rs), [shapely](https://shapely.readthedocs.io/en/stable/), [pyogrio](https://pyogrio.readthedocs.io/en/latest/introduction.html), [geoarrow-python](https://geoarrow.org/geoarrow-python/main/index.html), and [pyproj](https://pyproj4.github.io/pyproj/stable/index.html) for I/O and processing.

Spatial polars uses [lonboard](https://developmentseed.org/lonboard/latest/) for displaying geospatial data on an interactive map

Spatial polars is mostly just glue code connecting the work of others to bring spatial funcitonality to polars.

## Installation
Spatial polars can be installed from pypi
```title="Install with no optional dependencies"
pip install spatial-polars
```
```title="Install with dependencies for displaying data on a lonboard map"
pip install spatial-polars[lonboard]
```
```title="Install with dependencies to run examples in this guide (installs geodatasets)"
pip install spatial-polars[examples]
```
```title="Install with all optional dependencies"
pip install spatial-polars[lonboard, examples]
```

## Lazily access geospatial data
Spatial polars [scan_spatial](io.md#spatial_polars.io.scan_spatial) function will scan geoparquet files and any other data source [supported by pyogrio](https://pyogrio.readthedocs.io/en/latest/supported_formats.html) and return a [polars LazyFrame](https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html#). A [read_spatial](io.md#spatial_polars.io.read_spatial) function is also provided which simply wraps scan_spatial with a [.collect(streaming=True)][polars.LazyFrame.collect] at the end to return a polars dataframe. The scan_spatial function was the reason this package was created, it is much preferred over the read_spatial function for [the same reasons that polars recommends](https://docs.pola.rs/user-guide/concepts/lazy-api/) using the lazy API over the eager API.

## Geometry column
When spatial polars reads data from a spatial data source, the geometries are stored in a [polars struct](https://docs.pola.rs/user-guide/expressions/structs/) named "geometry" with two fields.  
<div class="annotate" markdown>
  * **wkb_geometry** field: A polars binary series for the geometry of each feature as [WKB](https://libgeos.org/specifications/wkb/) 
  * **crs** field: A polars categorical series for the [coordinate reference system as WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_coordinate_reference_systems) (1)
</div>
  
1. Using a categorical makes the series' RAM consumption very small.


Storing the geometries in this manner has an advantage over using a polars binary field holding [EWKB](https://libgeos.org/specifications/wkb/#extended-wkb), because this allows spatial polars to work with custom projections which do not have a SRID, without a need to store custom SRID codes/CRS definition elsewhere.

!!! warning "Warning about CRS"
    All geometries in a single column are expected to have the same CRS.  Currently there is nothing enforcing or validating all the geometries use the same CRS.

!!! note "Mixed geometry types in a single series"
    Spatial polars allows you to intermix geometry types (eg. points and lines) in the same geometry column.  Attempting to write a dataframe with a geometry column that has mixed geometry types may produce an error if the format is not capable of handling more than one geometry type.

## Spatial Expressions
Many expressions are included which work with the geometry struct.  The expressions all work in a similar manner

  1. Converts the polars series to a numpy array of WKB
  2. Converts the array of WKB to shapely geometry objects
  3. Uses shapely to do the spatial operation
  4. Depending on the result of the shapely function:
    * Result is an array of geometries: the result will be converted back to WKB and stored in a struct with the same CRS as the input.  
    * Result is **not** an array of geometries: the result will be an appropriately typed polars series.

Spatial polars expressions can be accesssed in two ways:  

```py title="Using the .spatial namespace"
df.with_columns(
    pl.col("geometry").spatial.buffer(10) # (1)!
)
```

1. This feels natural to polars users, and is totally usable, but typehints **will not** be avaliable in your IDE.

Or directly from the `SpatialExpr` class:
```py  title="Using the SpatialExpr class"
df.with_columns(
    SpatialExpr(pl.col("geometry")).buffer(10) # (1)!
)
```

1. This feels less natural to polars users, but typehints **will** be avaliable in your IDE.


### Spatial expressions which use more than geometry
Expressions in polars require a single column as the input. For computations involving two geometries, if the computation should be applied to the geometries in the column and a single other geometry, that geometry can be supplied to the expression as a scalar. [See geometry column and scalar geometry input expressions for more details.](examples/expressions/geometry_column_and_scalar_geometry_input_expressions.md)

```py  title="Spatial Expression with column of geometries and a scalar geometry"
my_point = shapely.Point(0,0)
df.with_columns(
    pl.col("geometry").spatial.distance(my_point).alias("dist_to_my_point") # (1)!
)
```

1. This will compute the distance of all the geometries in the `geometry` column of the dataframe to `my_point`

However if the computation needs to be run between two geometries each coming from different column in the dataframe, the two geometry struct columns must be placed into a single struct which is then supplied to the spatial polars expression. [See two geometry column input expressions for more details.](examples/expressions/two_geometry_column_input_expressions.md)

```py  title="Spatial Expression with two columns of geometries"
df.with_columns( # (1)!
    pl.struct(
        c.geometry, 
        c.geometry_other
    ).spatial.distance().alias("dist_to_other") # (2)!
)
```

1. This dataframe has two geomtry columns, one named `geometry` and another named `geometry_other`. 
2. wrapping the two geomtry columns in a struct and calling the `.spatial.distance` expression from the struct will compute the distance between each pair of geometries in the dataframe row-wise.

## Motivation
Spatial polars was motivated by interest in [polars IO plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/), and wanting to be able to easily read data from geopackages and GPX files (1) into a polars dataframe.
{ .annotate }

1.  GPX files are not offically supported by pyogrio, but seems to work on my end.

## Thank you!
This project would not be possible without all the work from the maintainers/contributors of of all the packages it's built on listed above, along with all the code they're built on, and inspired by.
