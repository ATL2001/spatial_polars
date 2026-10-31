---
draft: false 
date: 2026-10-31 
categories:
  - wkb
  - rust
  - expression
---

# That Escalated Quickly

## tldr
I've started dabbling with polars plugin expressions written in rust to do some stuff instead of using shapely or using other polars expressions that parse WKB.

## How I Got Here
With v0.4 I started using polars expressions to parse WKB and return information about geometries.  Prior to v0.4, simple operations like getting the X value of a point involved passing the WKB over to shapely to create a shapely geometry object, and then asking shapely to return the X coordinate of the point.  Using polars to read the X coordinate from the WKB was giving a pretty substantial speed boost for these simple operations, and I wanted to try to implement more expressions with a similar pattern.  I had already implemented the [get_exterior_ring](../../SpatialExpr.md#spatial_polars.spatialexpr.GeometryProperties.get_exterior_ring) expression using polars expressions, and thought it would be neat if I could compute the area of a polygon using a similar approach.  After a fair bit of struggle, I gave up :cry:.  Implementing the logic to grab the exterior ring of a polygon with a polars expression was one thing, but then also computing the area from those coordinates, dealing with holes in polygons, not to mention what to do with multipolygons and geometry collections, it was simply too much.  That being said, I had gotten myself excited to see what I could do, so I didnt stop there.

## The Escalation
Once apoun a time(1) when polars first introduced expressions plugins, [I wrote one](https://github.com/ATL2001/polars_uuid4).  The plugin created random UUID4 values for your dataframe.  I didn't then, nor do I now, consider myself any sort of expert in UUID's so I didnt publish it to pypi(2), and in the grand scale of the pipelines I was writing at the time, saving a couple seconds generating UUIDs wasn't going to make much of a difference in the end, so I just stuck with my python based UUID code.  This foray into UUID generation is here simply to say, I'm not :100: new to writing rust, I've gone thru a bit of [the rust book](https://doc.rust-lang.org/book/) and created that and a few other little odds and ends, but never anything in a professional capacity or anything very big, and the stuff I have done... it's safe to say I'm rather (3) rusty on.  I decided to see what I could do with a proper polars plugin expression.
{ .annotate }

1. Almost three years ago(1)
    { .annotate }
    1. Where does the time go?
2. It did do that one task well in my non UUID expert opinion 
3. Pun incoming... 

## The Approach
Instead of just jumping in straight with the rust code, I started by implementing a few functions in python which parsed WKB bytes to extract the coordinates of each of the rings of a polygon, and compute the area of the rings, and subtract the rings which are holes from the first ring's area.  This approach allowed me to test a few things about my logic in a language I already know well.  

* I know how to properly parse all the coordinates from the exterior and interior rings
* I could implement an algorithm that computes the area from a set of coordinates
* I could easily use shapely to verify that my result was correct

Then with my working python WKB area function(1) I turned to google and asked it to convert my python code to rust.  I looked over the code, and then slapped it into a blank polars expression package from [Marco's excellent polars plugin tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/) and wired it all up.  Did it work? nope.  The code wouldn't compile, but after a bit of work, I got it going.  After I got it working for a polygon and a polygon with holes in it, I took on the task of making it work for multipolygons and gometry collections writing the rust code myself.
{ .annotate }

1. Big shout out to [this OGC 'OpenGIS® Implementation Standard for Geographic information - Simple feature access - Part 1: Common architecture ' doc section 8.2.8](https://docs.ogc.org/is/06-103r4/06-103r4.pdf) for a great run down on WKB formatting, and my old [Spatial Databases: with Application to GIS](https://www.amazon.com/dp/1558605886?lv=shuf&channelId=500&plpRedirect=mhFallback)(1) text book for the refresher in how to compute the area of a polygon!
    { .annotate }

    1. Not an affiliate link.. I still like this book still after 20 something years

## The Result
I wrote some code that generated 1 million convex hull polygons each formed from 10 random points.  Then I added the ploygons as WKB to a spatial polars dataframe and used the [.spatial.area()](../../SpatialExpr.md#spatial_polars.spatialexpr.Measurement.area) expression to compute the area.  With version 0.4.1 on my machine the area operation took 1.2 seconds.  With version 0.5.0b4 it only took .2 seconds, 6X faster!

## Where to Next?
Honestly not sure where this is all going at this point other than me learning more.  I'm pretty jazzed that the polars team is [making progress for geopolars](https://github.com/pola-rs/polars/pull/29987) so hopefully that takes off and hopefully I can lend a hand over there!