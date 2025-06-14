## Spatial Filter

Spatial polars expressions which produce a boolean series can be used in polars filter context.  If we want to limit the rows in our dataframe to just the rows in a certain geographic extent we can use the [.spatial.intersects()](../../SpatialExpr.md#spatial_polars.spatialexpr.Predicates.intersects) expression with a shapely geometry object to filter our dataframe to just those rows.

--8<-- "geodatasets_note.md"

!!! Warning "Use `mask` or `bbox` if possible"
    This example reads data, filters it, then displays it on a map.  If you only need data from a specific area in your workflow, using the `mask` or `bbox` parameters in the creation of the lazy/dataframe will result in a better performance, as the filter would be applied by pyogrio when reading the data.

```py title="Using .spatial.intersects() to filter a dataframe" hl_lines="25-27"
--8<-- "spatial_filter.py"
```

1. Read the nyc_earnings geodataset into a dataframe
2. Reproject the data to WGS84 so we can use longitude/latitude coordinates for our filter polygon
3. Create a shapely polygon we can use to filter the rows in the dataframe, this polygon covers all of Rikers Island
4. Use .spatial.intersects to filter the dataframe to return the rows that intersect our polygon
5. Use lonboard's viz function to make a map
6. Create the map's layer from our boundary polygon
7. Symbolize the polygon with no fill color, make the outline red, and make the outline 35 meters wide
8. Make a polygonlayer from our filtered dataframe
9. Add our filtered polygonlayer to our map

```python exec="on" result="text"
--8<-- "spatial_filter.py"
```

<div class="map">
  <iframe src="../spatial_filter.html" width="800" height="504"></iframe>
</div>