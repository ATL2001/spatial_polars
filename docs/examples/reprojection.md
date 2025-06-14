# Reprojection

Spatial polars is happy to reproject your data using [pyproj](https://pyproj4.github.io/pyproj/stable/api/crs/crs.html) in the exact same manner as [geopandas](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html) using the [.reproject](../SpatialExpr.md#spatial_polars.spatialexpr.SpatialExpr.reproject) expression.

--8<-- "geodatasets_note.md"

To reproject data into a new CRS you can use the `.spatial.reproject()` expression

```python title="Reprojecting data to a new CRS" hl_lines="26"
--8<-- "reprojection.py"
```

1. Scan the nyc_earnings geodataset 
    
    !!! Note
        
        This data is in the USA_Contiguous_Albers_Equal_Area_Conic projection

2. Select the geometry column and convert it to shapely geometry objects
3. Collect the first row of the dataframe and grab the item from the dataframe since it only has one column and one row
4. Scan the same dataset as above
5. Select the geometry column reproject it to WGS84 and convert it to shapely geometry objects 
6. Collect the first row of the dataframe and grab the item from the dataframe since it only has one column and one row

```python exec="on" result="text"
--8<-- "reprojection.py"
```