# Reading Spatial Data

Spatial polars provides an [IO plugin](https://docs.pola.rs/user-guide/plugins/io_plugins/) that allows us to scan a spatial data into a lazyframe using the [scan_spatial](../io.md#spatial_polars.io.scan_spatial)  function.  The `scan_spatial` function uses [pyogrio](https://pyogrio.readthedocs.io/en/latest/introduction.html) under the hood to read the data, so [anything it can read](https://pyogrio.readthedocs.io/en/latest/supported_formats.html), spatial polars can read.

--8<-- "geodatasets_note.md"

```py title="Reading Spatial Data Setup"
import zipfile # (1)!

import geodatasets # (2)!
import polars as pl # (3)!
import shapely # (4)!

from spatial_polars import scan_spatial, read_spatial # (5)!
```

1. This is just so we can peek inside a zip file it's unnecessary for a typical workflow
2. See note above about geodatasets, also not needed for a normal workflow
3. yup, we need polars!
4. You may or may not need shapely imported, we need it for the mask example below to make a shapely polygon.  If you aren't using it directly you wont need to import it.
5. The scan_spatial function is where the spatial polars magic starts, and read_spatial is helpful too

We'll start by downloading the geoda.nyc_earnings dataset.

The data from geodatasets is zipped, but pyogrio has no issue reading directly out of a zip file, so there isn't any need to unzip anything first.  

Just to show what's inside the nyc_earnings dataset we'll just print out the contents of the zip file and see that there's a shapefile with all it's associated files.

```py title="looking inside the geoda.nyc_earnings zip file"
nyc_earnings_path = geodatasets.get_path("geoda.nyc_earnings")
with zipfile.ZipFile(nyc_earnings_path, "r") as zip_file:
    for file_name in zip_file.namelist():
        print(file_name)
```

``` { .yaml .no-copy }
NYC Area2010_2data.dbf # (1)!
__MACOSX/
__MACOSX/._NYC Area2010_2data.dbf
NYC Area2010_2data.prj # (2)!
__MACOSX/._NYC Area2010_2data.prj
NYC Area2010_2data.shp # (3)!
__MACOSX/._NYC Area2010_2data.shp
NYC Area2010_2data.shx # (4)!
__MACOSX/._NYC Area2010_2data.shx
```

1. Our shapefile's dbf of our tabular information
2. Our shapefile's coordinate system information
3. Our shapefile's geometries
4. Our shapefile's file that links the geometries and tabular information

## scan_spatial()

To scan the data and produce a polars LazyFrame we use the [scan_spatial](../io.md#spatial_polars.io.scan_spatial) function.  The cell below will produce a LazyFrame of the nyc earnings shapefile.  When we collect the schema, we'll see that it's got a lot of columns, the last of which is "geometry", a struct of our shapefile's geometry stored as WKB and CRS information.

```py title="Scanning a zipped shapefile"
nyc_earnings_lf = scan_spatial(geodatasets.get_path("geoda.nyc_earnings"))
nyc_earnings_lf.collect_schema()
```

``` { .yaml .no-copy }
Schema([('STATEFP10', String),
        ('COUNTYFP10', Int64),
        ('TRACTCE10', String),
        ('BLOCKCE10', String),
...SNIP... # (1)!
        ('CE02_14', Int64),
        ('CE03_14', Int64),
        ('geometry',
         Struct({'wkb_geometry': Binary, 'crs': Categorical(ordering='physical')}))]) # (2)!
```

1. There are 71 columns, but this documentation doesn't need to show them all :tongue:
2. This 'geometry' struct is the geometry from our data source, the 'wkb_geometry' field of the struct holds the geometry encoded as WKB, and the 'crs' field is the coordinate reference system for the geometries.

### Collecting the LazyFrame

To see the actual data, just like any other polars LazyFrame, we can collect the LazyFrame.

```py title="Collecting a LazyFrame into a DataFrame"
print(nyc_earnings_lf.collect(engine="streaming"))
```

``` { .yaml .no-copy }
shape: (108_487, 71) # (1)!
┌───────────┬────────────┬───────────┬───────────┬───┬─────────┬─────────┬─────────┬───────────────┐
│ STATEFP10 ┆ COUNTYFP10 ┆ TRACTCE10 ┆ BLOCKCE10 ┆ … ┆ CE01_14 ┆ CE02_14 ┆ CE03_14 ┆ geometry      │
│ ---       ┆ ---        ┆ ---       ┆ ---       ┆   ┆ ---     ┆ ---     ┆ ---     ┆ ---           │
│ str       ┆ i64        ┆ str       ┆ str       ┆   ┆ i64     ┆ i64     ┆ i64     ┆ struct[2]     │
╞═══════════╪════════════╪═══════════╪═══════════╪═══╪═════════╪═════════╪═════════╪═══════════════╡
│ 36        ┆ 5          ┆ 051600    ┆ 5011      ┆ … ┆ 0       ┆ 0       ┆ 0       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 5          ┆ 030000    ┆ 4003      ┆ … ┆ 22      ┆ 38      ┆ 51      ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 5          ┆ 040900    ┆ 1001      ┆ … ┆ 1       ┆ 3       ┆ 0       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 5          ┆ 040900    ┆ 2000      ┆ … ┆ 327     ┆ 552     ┆ 733     ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 5          ┆ 041100    ┆ 1000      ┆ … ┆ 0       ┆ 0       ┆ 0       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ …         ┆ …          ┆ …         ┆ …         ┆ … ┆ …       ┆ …       ┆ …       ┆ …             │
│ 36        ┆ 119        ┆ 002201    ┆ 2018      ┆ … ┆ 2       ┆ 1       ┆ 2       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
...
│ 36        ┆ 119        ┆ 000900    ┆ 1016      ┆ … ┆ 15      ┆ 21      ┆ 55      ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
└───────────┴────────────┴───────────┴───────────┴───┴─────────┴─────────┴─────────┴───────────────┘
```

1. That's 108K rows and 71 columns

### Optimizations

Like all the polars provided `scan_*` functions, scan spatial will make use of [polars optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/) when the query is collected.  This can lead to dramatic speed improvements:rocket:, and reduce RAM consumption.


```py title="Collecting a LazyFrame into a DataFrame with predicate/projection pushdown"
nyc_earnings_df = nyc_earnings_lf.filter(
    pl.col("COUNTYFP10") == 119 # (1)!
).select(
    pl.col("COUNTYFP10"), 
    pl.col("NAME10"),  # (2)!
    pl.col("geometry")
).collect(engine="streaming") # (3)!
print(nyc_earnings_df)
```

1. This filter will limit the rows to just the ones where COUNTYFP10 is 119
2. This will limit the columns that are read from the shapefile to just these three
3. Give us a DataFrame that has only the rows we filtered on, and just the columns we selected

``` { .yaml .no-copy }
shape: (15_081, 3) # (1)!
┌────────────┬────────────┬─────────────────────────────────┐
│ COUNTYFP10 ┆ NAME10     ┆ geometry                        │
│ ---        ┆ ---        ┆ ---                             │
│ i64        ┆ str        ┆ struct[2]                       │
╞════════════╪════════════╪═════════════════════════════════╡
│ 119        ┆ Block 1010 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 4005 ┆ {b"\x01\x03\x00\x00\x00\x02\x0… │
│ 119        ┆ Block 3023 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 3016 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 4008 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ …          ┆ …          ┆ …                               │
│ 119        ┆ Block 2018 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 2017 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 2018 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 1005 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
│ 119        ┆ Block 1016 ┆ {b"\x01\x03\x00\x00\x00\x01\x0… │
└────────────┴────────────┴─────────────────────────────────┘
```

1. Just a few rows and columns (remember, the shapefile has 108K rows and 71 columns).  Note that all the rows have COUNTYFP10 = 119 and obviously we only have three of the 71 columns

!!! Note "'Missing' parameters"

    Users that are familiar with pyogrio (or geopandas), may wonder why spatial polars has no "columns", "where", "fids", or "sql" parameters on the scan_spatial function.  These parameters are used by pygrio to limit the columns and rows that are read from a datasource.  Spatial polars operates differently.  Under the covers it does use the columns parameter from pyogrio, but instead of the user explicitly stating the columns in the scan_spatial function, polars will pass the columns needed for your query to pyogrio for you when it collects your query.  The where clause is different, polars expressions do not relate 1:1 to a where clause, so when the data is fetched by polars using the scan_spatial function, it will apply the polars predicate to each batch of records pyogrio reads before passing the records to the rest of the polars pipeline.
    
### The `bbox` Parameter

The `scan_spatial` function also has the ability to apply a spatial filter to the input data to only read in features that intersect a specific area.  The bounding box' extent can be provided to the `bbox` parameter as a tuple of floats for the xmin, ymin, xmax, ymax.

```py title="Spatial filter with the `bbox` parameter"
nyc_earnings_lf2 = scan_spatial(
    geodatasets.get_path("geoda.nyc_earnings"), 
    bbox=(1816900, 639414, 1817000, 639668) # (1)!
)
print(nyc_earnings_lf2.collect(engine="streaming"))
```

1. Only give us the rows that intersect this bounding box when we collect.  These corrdinates are just some numbers chosen at random which produce a small number of rows in the resulting dataframe.

``` { .yaml .no-copy }
shape: (4, 71) # (1)!
┌───────────┬────────────┬───────────┬───────────┬───┬─────────┬─────────┬─────────┬───────────────┐
│ STATEFP10 ┆ COUNTYFP10 ┆ TRACTCE10 ┆ BLOCKCE10 ┆ … ┆ CE01_14 ┆ CE02_14 ┆ CE03_14 ┆ geometry      │
│ ---       ┆ ---        ┆ ---       ┆ ---       ┆   ┆ ---     ┆ ---     ┆ ---     ┆ ---           │
│ str       ┆ i64        ┆ str       ┆ str       ┆   ┆ i64     ┆ i64     ┆ i64     ┆ struct[2]     │
╞═══════════╪════════════╪═══════════╪═══════════╪═══╪═════════╪═════════╪═════════╪═══════════════╡
│ 36        ┆ 119        ┆ 014701    ┆ 1010      ┆ … ┆ 0       ┆ 0       ┆ 0       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 119        ┆ 014701    ┆ 1022      ┆ … ┆ 2       ┆ 1       ┆ 2       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 119        ┆ 014701    ┆ 1006      ┆ … ┆ 16      ┆ 19      ┆ 28      ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 119        ┆ 014701    ┆ 1021      ┆ … ┆ 4       ┆ 4       ┆ 9       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
└───────────┴────────────┴───────────┴───────────┴───┴─────────┴─────────┴─────────┴───────────────┘
```

1. Just a 4 rows within that little box (remember, the shapefile has 108K rows and 71 columns)

### The `mask` Parameter

We could use those same coordinates in the bbox example as a shapely polygon to get the same results thru the mask parameter.  If we had a shapely polygon that was a non-rectangle shape that would be ok also.

```py title="Spatial filter with the mask parameter"
mask_polygon = shapely.Polygon( # (1)!
    (
        (1816900, 639414),
        (1816900, 639668),
        (1817000, 639668),
        (1817000, 639414),
        (1816900, 639414),
    )
)
nyc_earnings_lf3 = scan_spatial(
    geodatasets.get_path("geoda.nyc_earnings"), 
    mask=mask_polygon # (2)!
)
print(nyc_earnings_lf3.collect(engine="streaming"))
```

1. Make a polygon with the same coordinates as the bbox example above
2. Use the polygon to mask the reading of the data from the shapefile


``` { .yaml .no-copy }
shape: (4, 71) # (1)!
┌───────────┬────────────┬───────────┬───────────┬───┬─────────┬─────────┬─────────┬───────────────┐
│ STATEFP10 ┆ COUNTYFP10 ┆ TRACTCE10 ┆ BLOCKCE10 ┆ … ┆ CE01_14 ┆ CE02_14 ┆ CE03_14 ┆ geometry      │
│ ---       ┆ ---        ┆ ---       ┆ ---       ┆   ┆ ---     ┆ ---     ┆ ---     ┆ ---           │
│ str       ┆ i64        ┆ str       ┆ str       ┆   ┆ i64     ┆ i64     ┆ i64     ┆ struct[2]     │
╞═══════════╪════════════╪═══════════╪═══════════╪═══╪═════════╪═════════╪═════════╪═══════════════╡
│ 36        ┆ 119        ┆ 014701    ┆ 1010      ┆ … ┆ 0       ┆ 0       ┆ 0       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 119        ┆ 014701    ┆ 1022      ┆ … ┆ 2       ┆ 1       ┆ 2       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 119        ┆ 014701    ┆ 1006      ┆ … ┆ 16      ┆ 19      ┆ 28      ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
│ 36        ┆ 119        ┆ 014701    ┆ 1021      ┆ … ┆ 4       ┆ 4       ┆ 9       ┆ {b"\x01\x03\x │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 00\x00\x00\x0 │
│           ┆            ┆           ┆           ┆   ┆         ┆         ┆         ┆ 1\x0…         │
└───────────┴────────────┴───────────┴───────────┴───┴─────────┴─────────┴─────────┴───────────────┘
```

1. Just the same 4 rows as the bbox example within that little mask polygon (remember, the shapefile has 108K rows and 71 columns)


### The layer Parameter

Reading from a data source which contains more than one table is accomplished with the `layer` parameter.

geoda.milwaukee1 from geodatasets is a geopackage.  Geopackages are SQLite databases which can contain more than one table.

```py title="Using the layer parameter"
layer_param_df = scan_spatial(
    geodatasets.get_path("geoda.milwaukee1"), 
    layer="wi_final_census2_random4" # (1)!
).collect(engine="streaming")
print(layer_param_df)
```

1. Because there is more than one table in the geopackage we need to tell scan_spatial which table to read from.

``` { .yaml .no-copy }
shape: (417, 35)
┌─────────────┬───────────┬─────────┬────────┬───┬──────────┬──────────┬────────┬──────────────────┐
│ FIPS        ┆ MSA       ┆ TOT_POP ┆ POP_16 ┆ … ┆ PCTBLACK ┆ PCTBLCK  ┆ polyid ┆ geometry         │
│ ---         ┆ ---       ┆ ---     ┆ ---    ┆   ┆ ---      ┆ ---      ┆ ---    ┆ ---              │
│ str         ┆ str       ┆ i32     ┆ i32    ┆   ┆ f32      ┆ f32      ┆ i16    ┆ struct[2]        │
╞═════════════╪═══════════╪═════════╪════════╪═══╪══════════╪══════════╪════════╪══════════════════╡
│ 55131430100 ┆ Milwaukee ┆ 5068    ┆ 1248   ┆ … ┆ 0.860631 ┆ 0.000987 ┆ 1      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55089610100 ┆ Milwaukee ┆ 8003    ┆ 1812   ┆ … ┆ 0.005959 ┆ 0.004373 ┆ 2      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55131410100 ┆ Milwaukee ┆ 4393    ┆ 1026   ┆ … ┆ 0.030012 ┆ 0.000455 ┆ 3      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55131400101 ┆ Milwaukee ┆ 7687    ┆ 1801   ┆ … ┆ 0.141892 ┆ 0.000781 ┆ 4      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55131420104 ┆ Milwaukee ┆ 5086    ┆ 1065   ┆ … ┆ 0.010384 ┆ 0.012584 ┆ 5      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ …           ┆ …         ┆ …       ┆ …      ┆ … ┆ …        ┆ …        ┆ …      ┆ …                │
│ 55079160201 ┆ Milwaukee ┆ 8476    ┆ 1619   ┆ … ┆ 0.973236 ┆ 0.040349 ┆ 413    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55133203900 ┆ Milwaukee ┆ 7705    ┆ 1771   ┆ … ┆ 0.0      ┆ 0.004932 ┆ 414    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55079160202 ┆ Milwaukee ┆ 6169    ┆ 1532   ┆ … ┆ 0.000759 ┆ 0.01232  ┆ 415    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55079160300 ┆ Milwaukee ┆ 7398    ┆ 1915   ┆ … ┆ 0.007703 ┆ 0.022979 ┆ 416    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55079150400 ┆ Milwaukee ┆ 1092    ┆ 212    ┆ … ┆ 0.739302 ┆ 0.0      ┆ 417    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
└─────────────┴───────────┴─────────┴────────┴───┴──────────┴──────────┴────────┴──────────────────┘
```

## read_spatial()

Additionally, spatial polars also has a [read_spatial](../io.md#spatial_polars.io.read_spatial) function which will simply call `scan_spatial` and then call `.collect(engine="streaming")` to return a dataframe with all the columns and rows of a datasource.

```py title="read_spatial"
read_df = read_spatial( 
    geodatasets.get_path("geoda.milwaukee1"), 
    layer="wi_final_census2_random4"
) # (1)!
print(read_df)
```

1. No need to collect anything, read_spatial does that for you

``` { .yaml .no-copy }
shape: (417, 35) # (1)!
┌─────────────┬───────────┬─────────┬────────┬───┬──────────┬──────────┬────────┬──────────────────┐
│ FIPS        ┆ MSA       ┆ TOT_POP ┆ POP_16 ┆ … ┆ PCTBLACK ┆ PCTBLCK  ┆ polyid ┆ geometry         │
│ ---         ┆ ---       ┆ ---     ┆ ---    ┆   ┆ ---      ┆ ---      ┆ ---    ┆ ---              │
│ str         ┆ str       ┆ i32     ┆ i32    ┆   ┆ f32      ┆ f32      ┆ i16    ┆ struct[2]        │
╞═════════════╪═══════════╪═════════╪════════╪═══╪══════════╪══════════╪════════╪══════════════════╡
│ 55131430100 ┆ Milwaukee ┆ 5068    ┆ 1248   ┆ … ┆ 0.860631 ┆ 0.000987 ┆ 1      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55089610100 ┆ Milwaukee ┆ 8003    ┆ 1812   ┆ … ┆ 0.005959 ┆ 0.004373 ┆ 2      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55131410100 ┆ Milwaukee ┆ 4393    ┆ 1026   ┆ … ┆ 0.030012 ┆ 0.000455 ┆ 3      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55131400101 ┆ Milwaukee ┆ 7687    ┆ 1801   ┆ … ┆ 0.141892 ┆ 0.000781 ┆ 4      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55131420104 ┆ Milwaukee ┆ 5086    ┆ 1065   ┆ … ┆ 0.010384 ┆ 0.012584 ┆ 5      ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ …           ┆ …         ┆ …       ┆ …      ┆ … ┆ …        ┆ …        ┆ …      ┆ …                │
│ 55079160201 ┆ Milwaukee ┆ 8476    ┆ 1619   ┆ … ┆ 0.973236 ┆ 0.040349 ┆ 413    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55133203900 ┆ Milwaukee ┆ 7705    ┆ 1771   ┆ … ┆ 0.0      ┆ 0.004932 ┆ 414    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55079160202 ┆ Milwaukee ┆ 6169    ┆ 1532   ┆ … ┆ 0.000759 ┆ 0.01232  ┆ 415    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55079160300 ┆ Milwaukee ┆ 7398    ┆ 1915   ┆ … ┆ 0.007703 ┆ 0.022979 ┆ 416    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
│ 55079150400 ┆ Milwaukee ┆ 1092    ┆ 212    ┆ … ┆ 0.739302 ┆ 0.0      ┆ 417    ┆ {b"\x01\x03\x00\ │
│             ┆           ┆         ┆        ┆   ┆          ┆          ┆        ┆ x00\x00\x01\x0…  │
└─────────────┴───────────┴─────────┴────────┴───┴──────────┴──────────┴────────┴──────────────────┘
```

1. Exactly the same results as `scan_spatial().collect(engine="streaming")` (1)
   { .annotate }

    1. Because that's exactly what it's doing :smile: 
