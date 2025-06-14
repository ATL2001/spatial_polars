## Expressions involving two geometry columns

Spatial polars has many expressions that can involve two geometry columns from the frame where we want to compute the results of the expression row wise between the geometries in two columns. Because polars expressions only operate on one column, both geometry columns need to first be added to a struct before we call the spatial expression to perform the computation.

--8<-- "geodatasets_note.md"

To demonstrate, we'll start with our lake_boundary_df dataframe from the [spatial join example](../spatial_join.md)

```py title="Creating a dataframe with two geometry struct columns"
--8<-- "two_geometry_column_input_expressions.py:setup"
```

!!! Note
    1. For details about what's happening here see [spatial join example](../spatial_join.md)

Currently in the lake_boundary_df, the lakes which cross a boundary are represented by two rows.  Each row has a column "geometry" with the geometry of the lake, and a column "geometry_boundary" with the geometry of the boundary for the different boundaries. There is no differentiation of which portion of the lake is in which boundary.

To determine which part of the lake is in which boundary, we can use the [.intersection()](../../SpatialExpr.md#spatial_polars.spatialexpr.SetOperations.intersection) expression to determine the portion of the lake that intersects the boundary.  If we wanted to find the intersection of ALL the lakes to a single other polygon we could use the `other` parameter of the `.intersection()` method similar to how we used [.distance() in the geometry column and scalar geometry input expression](geometry_column_and_scalar_geometry_input_expressions.md), but since we want to know where the lakes from the geometry column intersect the boundary from the geometry_boundary column in a row-wise manner, we will add both the geometry and geometry_boundary columns to a struct and ignore the `other` parameter.  Spatial polars will then compute the intersection of the lake with the boundary and return the geometry of the lake which intersects the geometry of the boundary for each row, essentially cutting the lakes where they cross the bounary.

```py title="Computing the intersection of lakes and boundaries" hl_lines="6-9"
--8<-- "two_geometry_column_input_expressions.py:intersection"
```

1. Filter the dataframe to the rows that are in USA and Canada (just to make the map we'll produce later show up in an area where we have a lot of lakes crossing a boundary)
2. Add the 'geometry' and 'geometry_boundary' columns to a struct
3. Use the .spatial.intersection expression on the struct to compute a polygon that is the common area of the lake's geometry and the boundary's geometry for each set of lake/boundary row-wise

    !!! Note 
        Becasue we added the 'geometry' column to the struct first and, did not alias the result of the expression, the result of the expression will overwrite the data in the geometry column.  in this case that's what we want, but if you need to preserve the original geometry column, you will need to alias the result of the expression.
        
4. Drop the geometry_boundary column (we dont need it anymore)
5. Pass the dataframe to the [.viz() function](../../SpatialFrame.md#spatial_polars.spatialframe.SpatialFrame.viz) with `auto_highlight=True` so when we move our mouse over a polygon on the map it will change color

<div class="map">
  <iframe src="../lake_boundary.html" width="800" height="504"></iframe>
</div>