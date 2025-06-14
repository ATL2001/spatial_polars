# PathLayer (lines)

In this notebook we will demonstrate using spatial polars to visualize spatial data on a [Lonboard](https://developmentseed.org/lonboard/latest/) map using a path layer to visualize the polylines.

--8<-- "geodatasets_note.md"

## Creating a PathLayer

The following cell will read the eea large rivers polyline dataset into a dataframe and create a Lonboard [PathLayer](https://developmentseed.org/lonboard/latest/api/layers/path-layer/)

```py title="Creating a pathlayer" hl_lines="8-12"
--8<-- "pathlayer.py"
```

1. Read the eea.large_rivers geodataset into a dataframe
2. Create a pathlayer from the dataframe
3. When we mouse over a point in the map, it will change color
4. Color all the lines blue
5. Make them always at least 5px wide on the 
6. Add layer to a map and display it

<div class="map">
  <iframe src="../pathlayer.html" width="800" height="388"></iframe>
</div>