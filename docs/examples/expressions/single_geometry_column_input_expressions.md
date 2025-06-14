## Expressions involving a single geometry column
Spatial polars has many expressions which perform one of shapely's operations or return a property from each geometry in the series.  

To use these expressions which are called from a single geometry column, simply call the expression from the geometry column of the dataframe.

--8<-- "geodatasets_note.md"

```python hl_lines="11-14"
--8<-- "single_geometry_column_input_expressions.py"
```

1. Scan some polygon data 
2. Compute [.area()](../../SpatialExpr.md#spatial_polars.spatialexpr.Measurement.area) for each polygon aliased as 'polygon_area'.
3. Compute polygon perimeter [.length()](../../SpatialExpr.md#spatial_polars.spatialexpr.Measurement.length) for each polygon aliased as 'polygon_perimeter'
4. Compute polygon extent/bounds [.bounds()](../../SpatialExpr.md#spatial_polars.spatialexpr.Measurement.bounds) for each polygon aliased as 'bounds'
5. Compute polygon [.centroid()](../../SpatialExpr.md#spatial_polars.spatialexpr.ConstructiveOperations.centroid) for each polygon aliased as 'centroid' 
6. Select the geometry column for our results
7. Collect the query to return a dataframe

As you can see in the results below, all these expressions [(and many others in the API)](../../SpatialExpr.md#spatial_polars.spatialexpr.SpatialExpr) are executed on a geometry column but depending on the expression different datatypes are returned

```python exec="on" result="text"
--8<-- "single_geometry_column_input_expressions.py"
```