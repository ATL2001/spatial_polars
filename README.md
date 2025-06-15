# Spatial Polars
A package that extends [polars](https://pola.rs) for working with geospatial data.

Spatial polars relies on [polars](https://pola.rs), [shapely](https://shapely.readthedocs.io/en/stable/), [pyogrio](https://pyogrio.readthedocs.io/en/latest/introduction.html), [geoarrow-python](https://geoarrow.org/geoarrow-python/main/index.html), and [pyproj](https://pyproj4.github.io/pyproj/stable/index.html) for I/O and processing.

Spatial polars uses [lonboard](https://developmentseed.org/lonboard/latest/) for displaying geospatial data on an interactive map

Spatial polars is mostly just glue code connecting the work of others to bring spatial funcitonality to polars.

## Documentation
Documentation can be found here: [https://atl2001.github.io/spatial_polars/](https://atl2001.github.io/spatial_polars/)

## Lazily access data
Spatial polars scan_spatial function will scan geoparquet files and any other data source [supported by pyogrio](https://pyogrio.readthedocs.io/en/latest/supported_formats.html) and return a polars lazy frame.  A read_spatial function is also provided which simply wraps scan_spatial with a .collect() at the end to return a polars dataframe. The scan_spatial function was the reason this package was created, it is much preferred over the read_spatial function for [the same reasons that polars recommends](https://docs.pola.rs/user-guide/lazy/using/) using the lazy API over the eager API.

## Geometry
When reading data from a spatial data source, the geometries are stored in a [polars struct](https://docs.pola.rs/user-guide/expressions/structs/), with one polars binary field holding the geometry of the feature as [WKB](https://libgeos.org/specifications/wkb/) and another polars categorical field which stores the [coordinate reference system as WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_coordinate_reference_systems). Storing the geometries in this manner has an advantage over using a polars binary field holding [EWKB](https://libgeos.org/specifications/wkb/#extended-wkb), because this allows spatial polars to work with custom projections which do not have a SRID, without a need to store custom SRID codes/CRS definition elsewhere.

All geometries in a single column are expected to have the same CRS, currently there is nothing enforcing or validating this.

Spatial polars allows you to intermix geometry types (eg. points and lines) in the same geometry column.  Attempting to write a dataframe with a geometry column that has mixed geometry types may produce an error if the format is not capable of handling more than one geometry type.

## Spatial Expressions
Many expressions are included which work with the geometry struct to convert the data in the polars series to a numpy array of WKB, then convert the array of WKB to shapely geometry objects, and then use shapely's functions to do the spatial operation, then if the result is an array of geometries, they will be converted back to WKB and stored in a struct with the same CRS as the input.  Shapely functions that return something which is not a geometry will result in an appropriately typed polars series.  Spatial polars expressions can be accesssed under the `.spatial` namespace or directly from the `SpatialExpr` class.  When accessing the expressions thru the `SpatialExpr` class type hints will be available.

### Spatial expressions which use more than one geometry
Expressions in polars require a single column as the input. For computations involving two geometries, if the computation should be applied to the geometries in the column and a single other geometry, that geometry can be supplied to the expression as a scalar. However if the computation needs to be run between two geometries each coming from different column in the dataframe, the two geometry struct columns must be placed into a single struct which is then supplied to the spatial polars expression.

## Motivation
Spatial polars was motivated by interest in polars IO plugins, and wanting to be able to easily read data from geopackages and GPX files into a polars dataframe.

## Thank you!
This project would not be possible without all the work from the maintainers/contributors of of all the packages it's built on listed above, along with all the code they're built on, and inspired by.
