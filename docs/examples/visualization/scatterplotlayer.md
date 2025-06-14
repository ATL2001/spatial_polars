# ScatterPlotLayer (points)

In this notebook we will demonstrate using spatial polars to visualize spatial data on a [Lonboard](https://developmentseed.org/lonboard/latest/) map usign a scatterplot layer to visualize the point data.

--8<-- "geodatasets_note.md"

## Creating a ScatterplotLayer

The following cell will read the geoda home sales point dataset into a dataframe and create a Lonboard [ScatterplotLayer](https://developmentseed.org/lonboard/latest/api/layers/scatterplot-layer/)

```py title="Creating a scatterplotlayer" hl_lines="7-13"
--8<-- "scatterplotlayer.py"
```

1. Read the geoda.home_sales geodataset
2. Create the scatterplotlayer
3. Use the values in the 'floors' column to set the color of the points in the layer
4. Use a categorical color mapping dictionary to set the color of the points
5. If the number of floors is 1 the point will be red, two floors will be green, and three will be blue
6. The number of meters for the point symbols on the map
7. The points will not have an outline drawn around them
8. Show the scatterplotlayer on a map

<div class="map">
  <iframe src="../scatterplotlayer.html" width="800" height="388"></iframe>
</div>

!!! Note
    
    If you're interested in using a continuous colormap instead of a categorical, head over to [the polygonlayer example](polygonlayer.md) to check out how we can do it.