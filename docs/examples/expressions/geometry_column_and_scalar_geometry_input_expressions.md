## Expressions involving a geometry column and a scalar geometry
Spatial polars has many expressions which can be called from a geometry column but need another geometry to perform the operation.

If the desired output is to compute the operation on all the geometries in the series along with another single geometry, we can call the expression from the geometry column in our dataframe and provide the scalar geometry to the expression

--8<-- "geodatasets_note.md"

In the example below we'll use the distance expression to compute the distance of the counties in thelower 48 states of the USA to the gateway Arch in St. Louis Mo.  Then to give some context to the numbers, we'll display the data on a Lonboard map symbolized by the distance we calculated.

```py title="Compute Distance To Scalar Geometry" hl_lines="16"
--8<-- "geometry_column_and_scalar_geometry_input_expressions.py"
```

1. The point we want to compute the distance to for each of the polygons in our dataframe
2. Scan some polygon data
3. Select the county name
4. Select the geometry column (so we can show the data on a map)
5. Compute the distance of each polygon in the dataframe to our point
6. Collect the query to create a dataframe
7. Create a lonboard polygon layer that's symbolized by the distance we computed, making the smaller numbers(closer to the arch) red, and larger ones (further from the arch) green
8. Add the layer to a map and show the results

<div class="map">
  <iframe src="../distance_to_arch.html" width="800" height="504"></iframe>
</div>