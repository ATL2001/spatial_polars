# Starting From a Polars Dataframe

Spatial polars has a few ways to create a dataframe with a geometry column in the appropriate format for spatial polars functionality to operate that do not involve reading directly from a spatial source.  If you have an existing dataframe with coordinates of points, or WKB/WKT, you're in luck!

```py title="setup"
--8<-- "starting_from_a_polars_df.py:setup"
```
```python exec="on" result="text" session="starting_from_a_polars_df"
--8<-- "starting_from_a_polars_df.py:setup"
```

## From Point Coords

Spatial polars provides a way to take an existing polars dataframe with columns of X/Y (and optionally Z) coordinates, and convert them to work with the spatial polars functionality.  We can use [SpatialFrame.from_point_coords](../SpatialFrame.md#spatial_polars.spatialframe.SpatialFrame.from_point_coords) to take the dataframe and convert the coordinates into a spatial polars geometry struct column.

In the example below we'll create a polars DataFrame with a 'Place' column, for the name of the place, along with x/y/z coordinates.

```py title="from point coords"  hl_lines="11"
--8<-- "starting_from_a_polars_df.py:from_point_coords"
```

1. Creating a polars dataframe with x/y/z coordinates 
2. SpatialFrame.from_point_coords takes the x/y/z coordinates and converts them to a struct column with the geometry as WKB and a CRS
    
    !!! Note 
    
        Because we are not specifying a `crs` WGS84:4326 is defaulted.

```python exec="on" result="text" session="starting_from_a_polars_df"
--8<-- "starting_from_a_polars_df.py:from_point_coords"
```

## From WKB

Spatial polars provides a way to take an existing polars dataframe with a column of WKB, and convert it to work with the spatial polars functionality.  We can use [SpatialFrame.from_WKB](../SpatialFrame.md#spatial_polars.spatialframe.SpatialFrame.from_WKB) to take the dataframe and convert the WKB into a spatial polars geometry struct column.

In the cell below we'll create a polars DataFrame with a 'Place' column, for the name of the place, along with a column containting WKB.

```py title="from wkb"  hl_lines="11"
--8<-- "starting_from_a_polars_df.py:from_wkb"
```

1. Creating a polars dataframe with column of WKB
2. SpatialFrame.from_WKB takes the wkb column and adds it to a struct column with the CRS
    
    !!! Note 
    
        Because we are not specifying a `crs` WGS84:4326 is defaulted.

```python exec="on" result="text" session="starting_from_a_polars_df"
--8<-- "starting_from_a_polars_df.py:from_wkb"
```

## From WKT

Spatial polars provides a way to take an existing polars dataframe with a column of WKT, and convert it to work with the spatial polars functionality. We can use [SpatialFrame.from_WKT](../SpatialFrame.md#spatial_polars.spatialframe.SpatialFrame.from_WKT) to take the dataframe and convert the WKT into a spatial polars geometry struct column.

In the cell below we'll create a polars DataFrame with a 'Place' column, for the name of the place, along with a column containting WKT.

```py title="from wkt" hl_lines="11"
--8<-- "starting_from_a_polars_df.py:from_wkt"
```

1. Creating a polars dataframe with column of WKT
2. SpatialFrame.from_WKT takes the wkt column, converts it to WKB and adds it to a struct column with the CRS
    
    !!! Note 
    
        Because we are not specifying a `crs` WGS84:4326 is defaulted.

```python exec="on" result="text" session="starting_from_a_polars_df"
--8<-- "starting_from_a_polars_df.py:from_wkt"
```
