---
draft: false 
date: 2026-09-12 
categories:
  - wkb
---

Prior to V0.4.0 spatial polars treated WKB in the geometry structs as a black box.  For all spatial expressions, the WKB was shipped over to shapely, converted to a shapely geometry object, then whatever operation you requested was performed on the shapely goemetry object, then the results were passed back to polars.  It was simple, and shapely's numpy ufuncs certainly aren't what I consider slow, it was all good!  Then [DuckDB](https://duckdb.org/docs/lts/core_extensions/spatial/overview) and [SedonaDB](https://sedona.apache.org/sedonadb/latest/) came in with their blazing speeds, and made me once again reconsider my concept of slow :turtle:.

## An Idea Enters :bulb:
A few weeks ago, I stumbled on the [binary expressions](https://docs.pola.rs/api/python/stable/reference/expressions/binary.html) documentation offered by polars.  I'm sure I've seen that page before, but for some reason when I was there, it clicked!  A lot of spatial operations(1) are relatively simple math problems.  Example: to know the distance between two points(2), we just need to use the pythagorean theorem, which is trivial to implement using polars expressions.  I had a hunch that if we were to use polars expressions to parse the binary to extract the X/Y coordinates of the points and then supply those coords to polars expressions to compute the distance, we could get the same answer as before, but hopefully quicker, because we wouldn't need to create an entire shapely geometry object just to do some simple math.  
{ .annotate }

1. Especially ones involving points
2. In cartesian space (which is what shapely does)

## Let's Do it!
Armed with the idea, I set forth to see what I could do.  I took a dataframe with a bunch of points, and wrote a bare bones expression to parse the X coordinate from the WKB. It worked super fast, read **WAY** faster than sending the WKB to shapely and asking shapely to give me the X coordinate. I quickly spun up another expression to return the Y coordinate and then another to compute the pythagorean theorem to determine the distance to an arbitrary other point, and finally if the distance was less than an input distance to emulate the [dwithin function of shapely](https://shapely.readthedocs.io/en/latest/reference/shapely.dwithin.html).  Once again it was overwhelmingly faster than the shapely based function, I was extatic :flushed:!  Excitement aside, I knew what I was prototyping with was only dealing with little endian ISO WKB.  The expression I ultimately wanted needed to be taking into account WKB with big endian byte order, little endian extended WKB, and big endian extended WKB(1), and would still need a fallback (at least for now) to shapely for non-point geometry types.  Although I knew I needed to write a lot more code, I figured with the parallel nature of polars it would be a bit slower to take into account all those possibilities, but was hopeful that it would still net a large perfomance gain.
{ .annotate }

1. At the time I wasn't even thinking about extended WKB with an embedded SRID... which would further complicate matters... more on that later :-/ 

## Make it Robust
I began chaining out a big [pl.when().then().otherwise() expression](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.when.html#polars.when) to handle the different byte orders and WKB flavors.  I set up an expression to parse the binary to determine which geometry types were in the series, if they were little endian points, I supplied the WKB to my dwithin expression for little endian points, big endian points went to another expression, otherwise use shapely to compute the distance.  Long story short, that didn't pan out.  There's a big ole warning in the `polars.when` docs that clearly states that all the expressions would be evaluated in parallel(1), so even though the data I was experimenting with was all points, I after adding all the different `.when().then()` chains and having the fallback to use shapely, it was still doing all the computations including the fallback to shapely even though it wasnt going to use that result since I had the little endian point operation first in my chain... It was slower than just using shapely.  I was sad, but I didnt give up!
{ .annotate }

1. I didn't honestly pay attention to the big warning note there... I'm the problem, it's me.

## Where to Now?
I deicded to use a different expression first to split the big/little endiain point WKB data into different fields of struct, and then the non-points to a third field of a struct, so each row would have one of the three fields populated leaving the other two fields null. I passed the field of little endian WKB to one expression, the big endian to another, and the non-point geometry types field would go to shapely for the computation, then coalescing the results for each field in the struct.  In my experimentation with only points, since the data being passed to shapely was all None, shapely returned None back as a result super fast(1), and the little/big endian expressions were being run quickly, so overall I was getting much better performance(2) without touching the public API(3).  
{ .annotate }

1. I'm guessing it's got some sort of no-op shortcut when you give it a None to work from, it was almost instant on 6M points
2. Not as fast as just having only one expression which only worked for the little endian points, but still loads faster than the shapely method
3. I didnt want to make some sort of "only_points_dwithin" function or something like that, that feels complicated and confusing to me

## Computative Results
I pushed the updates up to github/pypi as V0.4.0, and asked the crew over at spatial bench if they'd be willing to kick off the github action to run that benchmark suite so I could see the results there.  Previously Q1 of the spatial bench benchmark, which uses the dwithin function, was taking spatial polars in the [3.8-3.9s range](https://github.com/apache/sedona-spatialbench/actions/runs/34016195606). Now running spatial polars v0.4.0 the exact same query, producing the same results only [took 0.82s](https://github.com/apache/sedona-spatialbench/actions/runs/34282584252) :rocket:  that's still a fair bit slower than DuckDB or SedonaDB, but it's not totally getting left in the dust anymore!


## Follow On
Overall I'm pretty jazzed :trumpet: about the results, and think I'll mess around with adding some more polars binary based expressions in the future. The more I was thought about it though, dealing with the different byte order and WKB flavors seems like a lot of overhead.  Not to mention I also woke up out of nowhere one night this week realizing that I hadn't accounted for embedded SRIDs, so results coming from extended WKB with an embedded SRID are wrong(1).  In version 0.4.1 I've decided to take all WKB input to the geometry series, and if it's **not** already little endian ISO, it will be converted to it, which resolves that SRID bug :bug:.  This will also ease the maintenance burden of all the different varieties of WKB, and shorten the when.then.otherwise chains, so theoretically we should get some more performance, provided the input data is already little endian ISO.
{ .annotate }

1. I'm being :100: serious... who thinks of this stuff in their sleep??? ... me I guess... :shrug:
