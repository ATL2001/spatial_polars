# PolygonLayer (polygons)

In this notebook we will demonstrate using spatial polars to visualize spatial data on a [Lonboard](https://developmentseed.org/lonboard/latest/) map using a polygon layer to visualize the polygons.

--8<-- "geodatasets_note.md"

## Creating a PolygonLayer

The following cell will read the nyc earnings dataset into a dataframe and create a Lonboard [PolygonLayer](https://developmentseed.org/lonboard/latest/api/layers/polygon-layer/)

I don't know much about the dataset but my understanding is that the CE03_* columns contain the number of jobs in within the polygon that earn more than $3333/month and all the polygons are in the New York City area, and it looks like "COUNTYFP10"==61 just gives us the polygons in Manhattan.  The content of the data isn't super critical, we're here just to show how we can use spatial polars to make a lonboard polygon layer that looks really nice.

```py title="Creating a polygonlayer" hl_lines="15-22"
--8<-- "polygonlayer.py"
```

1. Scan the geoda.nyc_earnings geodataset
2. Filter to just the rows in manhattan 
3. Create a polygonlayer
4. Use the values from the CE03_14 column to base our fill colors
5. The values in the column are continuous, not discreete categories, so we will use `fill_cmap_type="continuous"` 
6. [Palettable's](https://jiffyclub.github.io/palettable/) RdYlGn_11 color map will be used for the colors of the polygons
7. The values in the column are not all between 0-1 so we will use `fill_normalize_cmap_col=True` (True is the default, so we don't really need to specify it, but it's important so I wanted to call attention to it) which will scale a copy of the numbers in the column to be between 0-1 before using Lonboard's [apply_continuous_cmap](https://developmentseed.org/lonboard/latest/api/colormap/#lonboard.colormap.apply_continuous_cmap) function to set the color of each polygon.  If our column contained values that were all between 0-1 we could have used `fill_normalize_cmap_col=False`.
8. Use the "CE03_14" column to base our elevations of our polygons, so the different polygons will stick out of the map corresponding to the values in that column of our dataframe, Specifying the column name for the elevation will set the `extruded` property on the layer to be True.  
9. To make the edges of extruded polygons look really sharp, we'll set `wireframe=True`.
10. display the polygonlayer on a map

Because the data uses a coordinate reference system other than EPSG:4326, Lonboard will emit a warning that our input data is going to be re-projected to EPSG:4326.  To avoid that warning we could have reprojected the data in the dataframe with the spatial.reproject() expression on the geometry column before creating the PolygonLayer.

<div class="map">
  <iframe src="../polygonlayer.html" width="800" height="388"></iframe>
</div>

!!! Tip
    
    Right clicking, and moving your mouse on the lonboard map will allow you to tilt the view angle and really make those buildings look neat!

!!! Note
    If you're interested in using a categorical colormap instead of a continuous, head over to [the scatterplotlayer example](scatterplotlayer.md) to check out how we can do it.