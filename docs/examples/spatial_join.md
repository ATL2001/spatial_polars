# Spatial Joins

Spatial polars can perform a [spatial join](../SpatialFrame.md#spatial_polars.spatialframe.SpatialFrame.join) to join two dataframes based on geomtric predicate.  

!!! warning "Not lazy"
    Spatial joins are currently only implemented for DataFrames they are not yet available for LazyFrames.

!!! note "Thanks to Natural Earth"
    This example reads zipped data from [Natural Earth](https://www.naturalearthdata.com), big thanks to them for putting this data out there for us to use!

    Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.

To demonstrate how we can join data together from two dataframes spatially using spatial polars, we'll join some lake polygons with some administrative boundaries to see which lakes are in which countries.

```py title="Spatial Join" hl_lines="21-26"
--8<-- "spatial_join.py"
```

1. Reading the lakes with only the lake's name and geometry
2. Reading the boundaries with only the country's name (SOVEREIGNT) and geometry
3. Starting with the lake_df we'll start our spatial join
4. Specifying to join the lakes to this boundary_df
5. We'll use an inner join so as to only return rows for lakes that actually intersect a boundary.  If a lake does not intersect a boundary polygon we won't have a row for that lake in our output dataframe. Likewise, if a boundary doesn't intersect a lake, the resulting dataframe won't have a row for that boundary.

    !!! note
        This could have been left off, because `how='inner'` is the default

6. Use the 'intersects' spatial predicate so if any part of the lake shares any space with the boundary we'll join the lake to the boundary.  Since we've specified an inner join, if a lake intersects more than one boundary, we'll get more than one row for the lake since it's joined to more than one boundary.

    !!! note
        This could have been left off, because `predicate='intersects'` is the default

7. Since the name of the geometry struct is 'geometry' in both of our dataframes, we will use the `on` parameter, if we wanted to use a different column name for each of the dataframes we could use the `left_on` or `right_on` parameters. 

    !!! note
        This could have been left off, because `on='geometry'` is the default

8. Since we're joining the dataframes with a common column name (geometry), a suffix must be applied to the columns of the right dataframe that have names that exist in the left fram, because we can't have two columns with the same name.  we'll use "_boundary" as the suffix to clarify that the geometry of the right frame came from the boundaries dataframe.
9. Selecting the columns to make the lake name and SOVEREIGNT columns show up before the lake and boundary geometry columns.
10. Sort by the lake name just to make the results look nice in our output dataframe.

```python exec="on" result="text"
--8<-- "spatial_join.py"
```

Of the original 24 lakes and 177 bounaries, we have 36 rows now, beacause there are a few that cross the borders of the boundaries and were joined to more than one.  Lake Victoria is one of these lakes, it intersects both Kenya and Uganda

The resulting dataframe has a row for each lake name/geometry from the lakes dataframe and the SOVEREIGNT and geometry_boundary from the boundary df where the lake intersects an admin boundary.

Many GIS don't allow for multiple geometries on a single row, but spatial polars has no issue with this, and is actually [something that we can use to our advantage](../examples/expressions/two_geometry_column_input_expressions.md).