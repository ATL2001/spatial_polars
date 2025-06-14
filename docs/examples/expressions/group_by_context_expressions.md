# Group by Expressions

Spatial polars has expressions ([.intersection_all](../../SpatialExpr.md#spatial_polars.spatialexpr.SetOperations.intersection_all) and [.union_all](../../SpatialExpr.md#spatial_polars.spatialexpr.SetOperations.union_all)) that are designed to be used in a [group_by context](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/#group_by-and-aggregations).

--8<-- "geodatasets_note.md"

To demonstrate the usage of these functions, in the cell below we'll use polars to group some polygon data based on the `COUNTYFP10` column (an int column which holds a different code for each of the 9 different counties in the data). Then we'll compute the sum one column, count the number of rows in each group and compute the intersection of all the geometries of the polygons in the group, and show the data on a map.
    

```py title="Computing the union of all geometries in a group_by context" hl_lines="14"
--8<-- "group_by_context_expressions.py"
```

1. Scan the nyc_earnings dataset into a lazyframe.
2. Group the data by the `COUNTYFP10` column.
3. Sum the values in the CE03_14
4. Count the number of rows of the input data are in each group
5. Use the union_all expression to union all the geometries of the polygons that belong to each county.
6. Collect the query to create a dataframe
7. Make a polygon layer symbolized by the summed CE03_14 values 
8. Display the layer on a lonboard map.

<div class="map">
  <iframe src="../group_by_county.html" width="800" height="504"></iframe>
</div>
