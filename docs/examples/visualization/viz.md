# Vis

In this notebook we will demonstrate using spatial polars to visualize spatial data on a [Lonboard](https://developmentseed.org/lonboard/latest/) map using Lonboard's .vis function.


--8<-- "geodatasets_note.md"


Sometimes you just want to see stuff on a map to check out what the data looks like, you dont really care about the colors or any specifics, just show me something.  Lonboard's [viz](https://developmentseed.org/lonboard/latest/api/viz/) has you covered.  All we have to do is call  [.spatial.viz()](../../SpatialFrame.md/#spatial_polars.spatialframe.SpatialFrame.viz) and it will take the data in our dataframe and add it to a map, no need to fiddle with making a map and a layer and adding the layer to the map, super easy!  If you have specific parameters to set for the layer or map, they can be provided as dictionaries to the *_kwargs parameters for the layer type or the map.

In the example below we'll read the geoda.milwaukee1 dataset into a dataframe and the call `.spatial.viz()` from the dataframe, and it will give us a map showing some polygons around the Milwaukee area.  Big shout out to all my people over in Mukwonago (it's where ya wanna go!)

```py title="using .viz" hl_lines="6"
--8<-- "viz.py"
```

1. Read the geoda.milwaukee1 geodataset into a dataframe
2. Create a map from the dataframe using viz with no kwargs


<div class="map">
  <iframe src="../viz.html" width="800" height="388"></iframe>
</div>