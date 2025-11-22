---
draft: false 
date: 2025-11-21 
categories:
  - pyogrio
  - shapely
  - lonboard
  - spatial bench
---

# Hey look, a spatial polars blog! (or: why I made this)
I've been a huge fan of using polars for data engineering work since I started using it in early 2023.  I had read about how polars was faster than pandas (1) for various operations, and figured it was worth a try.  Like many polars users who have used pandas previously, I quickly fell in love with it's API, strong typing, and that it allows nulls in all of it's series no matter the datatype.  The funny part is, I'm not a programmer or data engineer by schooling, I'm actually a geography dork who like solving problems with data(2). I've found over the past 20 something years, many problems in the data analytics world can benefit greatly from a geospatial data/processes, and many problems that are encountered in the traditional GIS/geospatial space can greatly benefit from the tooling/ideas that exist from outside the GIS space.  That being said, polars has no native support for reading geospatial data formats.  At work I have written functions that read from some ESRI formats and produce polars dataframes, but when I'm at home working with spatial data for [personal projects](https://huggingface.co/spaces/ATL2001/hiking_club) I didn't have a seamless way to get data from formats like GPX or geopackages into polars.
{ .annotate }

1. I had been using :panda:s daily for 6 ish years at this point
2. I started programming out of the frustration that the tools I wanted to use didn't exist (or perhaps I was just looking in the wrong places for the tools I wanted :thinking:)

## IO plugins arrive
When expressions plugins for polars were introduced, I built [a simple expression plugin](https://github.com/ATL2001/polars_uuid4)(1), so when [polars 1.22.0](https://github.com/pola-rs/polars/releases/tag/py-1.22.0) came out with support for [IO plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/), I was immediately interested in seeing how the guts of the IO plugins worked.  The IO plugins section of the user guide made it very easy to understand what was needed.  I was aware of [pyogrio](https://pyogrio.readthedocs.io/en/latest/index.html) and how it was able to [produce a pyarrow RecordBatchReader](https://pyogrio.readthedocs.io/en/latest/api.html#pyogrio.open_arrow), and looking at the IO plugin guide it was obvious to me that they were a perfect fit for one another.  I learn things best by doing, so I decided I wanted to see if I could take pyogrio and feed the data into the polars IO plugin.  It took me a while to settle on how to structure the spatial data in the dataframes to make it easy to work with, performant, not require more RAM than necessary, and be able to work with custom coordinate systems, not just ones with a well known SRID(2).  What I settled on was using a polars struct column containing two fields, one to store a geometry as WKB, and another that contains the coordinate refrence system's WKT stored in the field as a categorical which makes the RAM for the CRS essentially negligible.  I proceeded to wire up the scan_spatial function in spatial polars and was very pleased at the speed at which I was able to read from all sorts of geospatial formats into a polars dataframe.
{ .annotate }

1. This never made it to the python package index, someday I may spruce it up and create a package from it.
2. I've encountered custom coordinate systems regularly in the past 2x years working with spatial data.

## Spatial data deserves Spatial stuff

Now that I had created a way to scan a spatial source into a polars lazyframe, I needed to have a way to use the spatial data as it was structured in the frame. After all, reading the data is only half the battle(1), if I couldn't process and display the data, it had limited use.  For operations, I knew I could rely on [shapely's numpy ufuncs](https://shapely.readthedocs.io/en/stable/index.html#what-is-a-ufunc) to come to the rescue to process spatial data quickly(2).  For visualizing the data, I turned to [lonboard](https://developmentseed.org/lonboard/latest/) because of it's ability to render large volumes of data dramatically faster than any other map viewer I've ever worked with.
{ .annotate }

1. ...or something like that...
2. The vectorized functions in shapely have fundamentally changed the way I approach processing spatial data :rocket:

## So why make a new package and not contribute to something else?

### Geopolars
[Geopolars](https://geopolars.org/latest/) is blocked, although after I started working on the code for this project, the polars team has stated ["We want to do this. We hope to get to this soon™"](https://github.com/pola-rs/polars/issues/9112#issuecomment-2879650887) which is totally awesome, and when that happens, I'm hopeful that I'll be able to contribute to that project, because I think it has a lot more potential than what I've got going here.

### Polars ST
[Polars ST](https://oreilles.github.io/polars-st/) is a working alternative to what I've created here, and it looks pretty nice:thumbsup:!  However the use of EWKB doesn't work for me and my desire to support coordinate reference systems which don't have an SRID, and I didn't feel that coming into a new project and saying something along the lines of "this is nice! can we restructure everything so it fits an edge case that I may run into someday?".

### I wanted to learn something new
Before this project I had never put anything on the python package index, and all this fancy github CI/CD stuff was totally new to me, and I really wanted to learn more about using [mkdocs](https://www.mkdocs.org)/[mkdocs-material](https://squidfunk.github.io/mkdocs-material) for creating documentation like this.  I figured what better way to learn it all than by doing it.

## Why now?
The funny part is that I started writing this blog article almost 5 months ago, but it simply hasnt been high on my todo list. Then, the apache sedona people released [sedonadb](https://sedona.apache.org/sedonadb/latest), a single node database engine that supports geospatial data. At the same time they also released [spatialbench](https://sedona.apache.org/spatialbench), a tool for assesing spatial query performance of various engines.  I was absolutely PUMPED to see what they had come up with in spatialbench.  Don't get me wrong, the query engine, sedonadb is pretty neat, it is SUPER fast :rocket: (1) but I still prefer the expressions api provided by polars/spatial polars(2)(3). Spatialbench includes benchmarks of the same queries executed on the same data with SedonaDB, DuckDB, and GeoPandas. While building spatial polars, I kept second guessing myself about the speed I was seeing comparing the results of the operations to geopandas (4). I kept thinking to myself, "it'd be really great if we had some sort of benchmarking stuff similar to [TPC-H](https://www.tpc.org/tpch) or [TPC-DS](https://www.tpc.org/tpcds) for comparing spatial queries" and then poof, there it was! They were thinking  the same thing as I was, AND THEY DID IT! :hands-raised:.  Now that we had a set of queries to compare the engines I set forth to see where spatial polars stacked up.  I had to make a few changes to spatial polars along the way (specifically around generating convex hulls from multiple geometries resulting from a group_by, and I didn't have any sort of KNN join capabilities in spatial polars before trying to create [Q12](https://sedona.apache.org/spatialbench/queries/#q12-find-five-nearest-buildings-to-each-trip-pickup-location-using-knn-join) (5)),  But I've got all 12 queries built using spatial polars, and I'm currently working with the maintainers of spatial bench to add the queries using spatial polars :smile:.  So I figured what better way to celebrate all this stuff than finally getting around to finishing up that introductory blog article! :tada:
{ .annotate }

1. A lot faster than what spatial polars offers, or ever will
2. As a developer/analyst/data engineer, I'm much more productive chaining out polars contexts/epressions than I ever was at composing SQL (1) 
{ .annotate }
    
    1. I wrote SQL successfuly for years, but since I was introduced to polars I've never looked back.

3. If you're looking for all out speed on a single node for geospatial queries, you're better off over there than here.
4. I've used geopandas a fair bit, but I certainly don't consider myself an authority, so I always had in the back of my mind that perhaps why I was seeing some dramatically faster results on some operations using spatial polars was just because I was doing GeoPandas wrong. :shrug:
5. I may have cheated on that KNN join, what I created only considers centroids of features... I'm not sure if the other pacakges consider the entire polygons or not.

## The Future
I don't honestly think this is a long term project. I'm very impressed by the geoarrow project, and think that when the polars extensions are released to the world, and geopolars can really get going, I'll try to help out there and make that project a success.  Although what I've created here is useful, I believe geopolars has a lot more potential than spatial polars, and I certainly don't have any interest in fragmenting the ecosystem more than I've potentially already done.