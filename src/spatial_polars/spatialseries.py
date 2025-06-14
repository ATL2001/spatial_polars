import polars as pl
import pyproj
import shapely

from polars import col as c

from .vendor import TransformerFromCRS, transform


@pl.api.register_series_namespace("spatial")
class SpatialSeries:
    def __init__(self, s: pl.Series) -> None:
        self._s = s

    def get_crs(self):
        """
        Returns the CRS WKT from a spatial series.
        """
        return self._s.struct.field("crs")[0]

    def to_shapely_array(self):
        """
        Returns a numpy array of shapely geometry objects from a spatial series.
        """
        geom_series = self._s.struct.field("wkb_geometry")
        return shapely.from_wkb(geom_series)

    @staticmethod
    def _to_spatialseries(wkb_array, crs_wkt):
        """
        Returns a polars struct series with a binary field for WKB geometries of the features and a categorical field that contains the CRS WKT.
        """
        result = (
            pl.DataFrame([wkb_array], schema={"wkb_geometry": pl.Binary}, strict=False)
            .with_columns(pl.lit(crs_wkt, dtype=pl.Categorical).alias("crs"))
            .select(pl.struct(c.wkb_geometry, c.crs).alias("geometry"))
            .to_series()
        )

        return result

    def reproject(self, crs_to):
        """
        Reprojects the SpatialSeries to a different CRS.
        """

        crs_from = self._s.spatial.get_crs()
        transformer = TransformerFromCRS(crs_from, crs_to, always_xy=True)

        s_arr = self.to_shapely_array()
        result = shapely.to_wkb(transform(s_arr, transformer.transform))

        crs = pyproj.CRS(crs_to)
        crs_wkt = crs.to_wkt()
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    # Measurement
    def area(self):
        """
        Compute the area of a (multi)polygon.
        """
        s_arr = self.to_shapely_array()
        result = shapely.area(s_arr)
        return result

    def distance(self, other):
        """
        Compute the Cartesian distance between two geometries.
        """
        s_arr = self.to_shapely_array()
        result = shapely.distance(s_arr, other)
        return result

    def bounds(self):
        """
        Compute the bounds (extent) of a geometry.

        For each geometry these 4 numbers are returned a struct: min x, min y, max x, max y.
        """
        s_arr = self.to_shapely_array()
        result = shapely.bounds(s_arr)
        return result

    def length(self):
        """
        Compute the length of a (multi)linestring or polygon perimeter.
        """
        s_arr = self.to_shapely_array()
        result = shapely.length(s_arr)
        return result

    def hausdorff_distance(self, other, densify=None):
        """
        Compute the discrete Hausdorff distance between two geometries.

        The Hausdorff distance is a measure of similarity: it is the greatest distance between any point in A and the closest point in B. The discrete distance is an approximation of this metric: only vertices are considered. The parameter 'densify' makes this approximation less coarse by splitting the line segments between vertices before computing the distance.
        """
        s_arr = self.to_shapely_array()
        if isinstance(other, pl.Series):
            other = other.to_shapely_array()
        result = shapely.hausdorff_distance(s_arr, other, densify)
        return result

    def frechet_distance(self, other, densify=None):
        """
        Compute the discrete Fréchet distance between two geometries.

        The Fréchet distance is a measure of similarity: it is the greatest distance between any point in A and the closest point in B. The discrete distance is an approximation of this metric: only vertices are considered. The parameter 'densify' makes this approximation less coarse by splitting the line segments between vertices before computing the distance.

        Fréchet distance sweep continuously along their respective curves and the direction of curves is significant. This makes it a better measure of similarity than Hausdorff distance for curve or surface matching.
        """
        s_arr = self.to_shapely_array()
        if isinstance(other, pl.Series):
            other = other.to_shapely_array()
        result = shapely.frechet_distance(s_arr, other, densify)
        return result

    def minimum_clearance(self):
        """
        Compute the Minimum Clearance distance.

        A geometry's "minimum clearance" is the smallest distance by which a vertex of the geometry could be moved to produce an invalid geometry.

        If no minimum clearance exists for a geometry (for example, a single point, or an empty geometry), infinity is returned.
        """
        s_arr = self.to_shapely_array()
        result = shapely.minimum_clearance(s_arr)
        return result

    def minimum_bounding_radius(self):
        """
        Compute the radius of the minimum bounding circle of an input geometry.
        """
        s_arr = self.to_shapely_array()
        result = shapely.minimum_bounding_radius(s_arr)
        return result

    # Predicates
    def has_z(self):
        """
        Return True if a geometry has Z coordinates.

        Note that for GEOS < 3.12 this function returns False if the (first) Z coordinate equals NaN.
        """
        s_arr = self.to_shapely_array()
        result = shapely.has_z(s_arr)
        return result

    def is_ccw(self):
        """
        Return True if a linestring or linearring is counterclockwise.

        Note that there are no checks on whether lines are actually closed and not self-intersecting, while this is a requirement for is_ccw. The recommended usage of this function for linestrings is is_ccw(g) & is_simple(g) and for linearrings is_ccw(g) & is_valid(g).
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_ccw(s_arr)
        return result

    def is_closed(self):
        """
        Return True if a linestring's first and last points are equal.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_closed(s_arr)
        return result

    def is_empty(self):
        """
        Return True if a geometry is an empty point, polygon, etc.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_empty(s_arr)
        return result

    def is_geometry(self):
        """
        Return True if the object is a geometry.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_geometry(s_arr)
        return result

    def is_missing(self):
        """
        Return True if the object is not a geometry (None).
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_missing(s_arr)
        return result

    def is_ring(self):
        """
        Return True if a linestring is closed and simple.

        This function will return False for non-linestrings.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_ring(s_arr)
        return result

    def is_simple(self):
        """
        Return True if the geometry is simple.

        A simple geometry has no anomalous geometric points, such as self-intersections or self tangency.

        Note that polygons and linearrings are assumed to be simple. Use is_valid to check these kind of geometries for self-intersections.

        This function will return False for geometrycollections.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_simple(s_arr)
        return result

    def is_valid(self):
        """
        Return True if a geometry is well formed.

        Returns False for missing values.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_valid(s_arr)
        return result

    def is_valid_input(self):
        """
        Return True if the object is a geometry or None.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_valid_input(s_arr)
        return result

    def is_valid_reason(self):
        """
        Return a string stating if a geometry is valid and if not, why.

        Returns None for missing values.
        """
        s_arr = self.to_shapely_array()
        result = shapely.is_valid_reason(s_arr)
        return result

    def crosses(self, other):
        """
        Return True if A and B spatially cross.

        A crosses B if they have some but not all interior points in common, the intersection is one dimension less than the maximum dimension of A or B, and the intersection is not equal to either A or B.
        """
        s_arr = self.to_shapely_array()
        result = shapely.crosses(s_arr, other)
        return result

    def contains(self, other):
        """
        Return True if geometry B is completely inside geometry A.

        A contains B if no points of B lie in the exterior of A and at least one point of the interior of B lies in the interior of A.

        Note: following this definition, a geometry does not contain its boundary, but it does contain itself. See contains_properly for a version where a geometry does not contain itself.
        """
        s_arr = self.to_shapely_array()
        result = shapely.contains(s_arr, other)
        return result

    def contains_properly(self, other):
        """
        Return True if geometry B is completely inside geometry A, with no common boundary points.

        A contains B properly if B intersects the interior of A but not the boundary (or exterior). This means that a geometry A does not "contain properly" itself, which contrasts with the contains function, where common points on the boundary are allowed.

        Note: this function will prepare the geometries under the hood if needed. You can prepare the geometries in advance to avoid repeated preparation when calling this function multiple times.
        """
        s_arr = self.to_shapely_array()
        result = shapely.contains_properly(s_arr, other)
        return result

    def covered_by(self, other):
        """
        Return True if no point in geometry A is outside geometry B.
        """
        s_arr = self.to_shapely_array()
        result = shapely.covered_by(s_arr, other)
        return result

    def covers(self, other):
        """
        Return True if no point in geometry B is outside geometry A.
        """
        s_arr = self.to_shapely_array()
        result = shapely.covers(s_arr, other)
        return result

    def disjoint(self, other):
        """
        Return True if A and B do not share any point in space.

        Disjoint implies that overlaps, touches, within, and intersects are False. Note missing (None) values are never disjoint.

        """
        s_arr = self.to_shapely_array()
        result = shapely.disjoint(s_arr, other)
        return result

    def equals(self, other):
        """
        Return True if A and B are spatially equal.

        If A is within B and B is within A, A and B are considered equal. The ordering of points can be different.
        """
        s_arr = self.to_shapely_array()
        result = shapely.equals(s_arr, other)
        return result

    def intersects(self, other):
        """
        Return True if A and B share any portion of space.

        Intersects implies that overlaps, touches, covers, or within are True.
        """
        s_arr = self.to_shapely_array()
        result = shapely.intersects(s_arr, other)
        return result

    def overlaps(self, other):
        """
        Return True if A and B spatially overlap.

        A and B overlap if they have some but not all points/space in common, have the same dimension, and the intersection of the interiors of the two geometries has the same dimension as the geometries themselves. That is, only polyons can overlap other polygons and only lines can overlap other lines. If A covers or is within B, overlaps won't be True.

        If either A or B are None, the output is always False.
        """
        s_arr = self.to_shapely_array()
        result = shapely.overlaps(s_arr, other)
        return result

    def touches(self, other):
        """
        Return True if the only points shared between A and B are on their boundaries.
        """
        s_arr = self.to_shapely_array()
        result = shapely.touches(s_arr, other)
        return result

    def within(self, other):
        """
        Return True if geometry A is completely inside geometry B.

        A is within B if no points of A lie in the exterior of B and at least one point of the interior of A lies in the interior of B.
        """
        s_arr = self.to_shapely_array()
        result = shapely.within(s_arr, other)
        return result

    def relate(self, other):
        """
        Return a string representation of the DE-9IM intersection matrix.
        """
        s_arr = self.to_shapely_array()
        result = shapely.relate(s_arr, other)
        return result

    def contains_xy(self, x, y):
        """
        Return True if the Point (x, y) is completely inside geom.

        This is a special-case (and faster) variant of the contains function which avoids having to create a Point object if you start from x/y coordinates.

        Note that in the case of points, the contains_properly predicate is equivalent to contains.

        See the docstring of contains for more details about the predicate.
        """
        s_arr = self.to_shapely_array()
        result = shapely.contains_xy(s_arr, x, y)
        return result

    def dwithin(self, other, distance):
        """
        Return True if the geometries are within a given distance.

        Using this function is more efficient than computing the distance and comparing the result.
        """
        s_arr = self.to_shapely_array()
        result = shapely.dwithin(s_arr, other, distance)
        return result

    def intersects_xy(self, x, y):
        """
        Return True if geom and the Point (x, y) share any portion of space.

        This is a special-case (and faster) variant of the intersects function which avoids having to create a Point object if you start from x/y coordinates.

        See the docstring of intersects for more details about the predicate.
        """
        s_arr = self.to_shapely_array()
        result = shapely.intersects_xy(s_arr, x, y)
        return result

    def equals_exact(self, other, tolerance):
        """
        Return True if the geometries are structurally equivalent within a given tolerance.

        This method uses exact coordinate equality, which requires coordinates to be equal (within specified tolerance) and in the same order for all components (vertices, rings, or parts) of a geometry. This is in contrast with the equals function which uses spatial (topological) equality and does not require all components to be in the same order. Because of this, it is possible for equals to be True while equals_exact is False.

        The order of the coordinates can be normalized (by setting the normalize keyword to True) so that this function will return True when geometries are structurally equivalent but differ only in the ordering of vertices. However, this function will still return False if the order of interior rings within a Polygon or the order of geometries within a multi geometry are different.
        """
        s_arr = self.to_shapely_array()
        result = shapely.equals_exact(s_arr, other, tolerance)
        return result

    def relate_pattern(self, other, pattern):
        """
        Return True if the DE-9IM relationship code satisfies the pattern.

        This function compares the DE-9IM code string for two geometries against a specified pattern. If the string matches the pattern then True is returned, otherwise False. The pattern specified can be an exact match (0, 1 or 2), a boolean match (uppercase T or F), or a wildcard (*). For example, the pattern for the within predicate is 'T*F**F***'.
        """
        s_arr = self.to_shapely_array()
        result = shapely.relate_pattern(s_arr, other, pattern)
        return result

    # Set operations
    def difference(self, other, grid_size=None):
        """
        Return the part of geometry A that does not intersect with geometry B.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.difference(s_arr, other, grid_size=grid_size))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def intersection(self, other, grid_size=None):
        """
        Return the geometry that is shared between input geometries.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.intersection(s_arr, other, grid_size=grid_size))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def intersection_all(self, grid_size=None):
        """
        Return the intersection of multiple geometries.

        This function ignores None values when other Geometry elements are present. If all elements of the given axis are None, an empty GeometryCollection is returned.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.intersection_all(s_arr, grid_size=grid_size))
        # intersection_all returns a scalar, so wrap it in a list
        return SpatialSeries._to_spatialseries([result], crs_wkt)

    def symmetric_difference(self, other, grid_size=None):
        """
        Return the geometry with the portions of input geometries that do not intersect.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.symmetric_difference(s_arr, other, grid_size=grid_size)
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def union(self, other, grid_size=None):
        """
        Merge geometries into one.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.union(s_arr, other, grid_size=grid_size))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def union_all(self, grid_size=None):
        """
        Return the union of multiple geometries.

        This function ignores None values when other Geometry elements are present. If all elements of the given axis are None an empty GeometryCollection is returned.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.union_all(s_arr, grid_size=grid_size))
        # union_all returns a scalar, so wrap it in a list
        return SpatialSeries._to_spatialseries([result], crs_wkt)

    # Constructive operations
    def boundary(self):
        """
        Return the topological boundary of a geometry.

        This function will return None for geometrycollections.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.boundary(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def buffer(
        self,
        distance,
        quad_segs=8,
        cap_style="round",
        join_style="round",
        mitre_limit=5.0,
        single_sided=False,
        return_shapely=False,
    ):
        """
        Compute the buffer of a geometry for positive and negative buffer distance.

        The buffer of a geometry is defined as the Minkowski sum (or difference, for negative distance) of the geometry with a circle with radius equal to the absolute value of the buffer distance.

        The buffer operation always returns a polygonal result. The negative or zero-distance buffer of lines and points is always empty.
        """
        s_arr = self.to_shapely_array()
        result = shapely.to_wkb(
            shapely.buffer(
                s_arr,
                distance,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                mitre_limit=mitre_limit,
                single_sided=single_sided,
            )
        )
        crs_wkt = self._s.spatial.get_crs()
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def offset_curve(
        self,
        distance,
        quad_segs=8,
        join_style="round",
        mitre_limit=5.0,
        return_shapely=False,
    ):
        """
        Return a (Multi)LineString at a distance from the object.

        For positive distance the offset will be at the left side of the input line. For a negative distance it will be at the right side. In general, this function tries to preserve the direction of the input.

        Note: the behaviour regarding orientation of the resulting line depends on the GEOS version. With GEOS < 3.11, the line retains the same direction for a left offset (positive distance) or has opposite direction for a right offset (negative distance), and this behaviour was documented as such in previous Shapely versions. Starting with GEOS 3.11, the function tries to preserve the orientation of the original line.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.offset_curve(
                s_arr,
                distance,
                quad_segs=quad_segs,
                join_style=join_style,
                mitre_limit=mitre_limit,
            )
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def centroid(self):
        """
        Compute the geometric center (center-of-mass) of a geometry.

        For multipoints this is computed as the mean of the input coordinates. For multilinestrings the centroid is weighted by the length of each line segment. For multipolygons the centroid is weighted by the area of each polygon.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.centroid(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def clip_by_rect(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """
        Return the portion of a geometry within a rectangle.

        The geometry is clipped in a fast but possibly dirty way. The output is not guaranteed to be valid. No exceptions will be raised for topological errors.

        Note: empty geometries or geometries that do not overlap with the specified bounds will result in GEOMETRYCOLLECTION EMPTY.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.clip_by_rect(s_arr, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def concave_hull(self, ratio=0.0, allow_holes=False):
        """
        Compute a concave geometry that encloses an input geometry.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.concave_hull(s_arr, ratio=ratio, allow_holes=allow_holes)
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def convex_hull(self):
        """
        Compute the minimum convex geometry that encloses an input geometry.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.convex_hull(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def delaunay_triangles(self, tolerance=0.0, only_edges=False):
        """
        Compute a Delaunay triangulation around the vertices of an input geometry.

        The output is a geometrycollection containing polygons (default) or linestrings (see only_edges). Returns an empty geometry for input geometries that contain less than 3 vertices.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.delaunay_triangles(
                s_arr, tolerance=tolerance, only_edges=only_edges
            )
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def segmentize(self, max_segment_length: float):
        """
        Add vertices to line segments based on maximum segment length.

        Additional vertices will be added to every line segment in an input geometry so that segments are no longer than the provided maximum segment length. New vertices will evenly subdivide each segment.

        Only linear components of input geometries are densified; other geometries are returned unmodified.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.segmentize(s_arr, max_segment_length))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def envelope(self):
        """
        Compute the minimum bounding box that encloses an input geometry.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.envelope(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def extract_unique_points(self):
        """
        Return all distinct vertices of an input geometry as a multipoint.

        Note that only 2 dimensions of the vertices are considered when testing for equality.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.extract_unique_points(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def build_area(self):
        """
        Create an areal geometry formed by the constituent linework of given geometry.

        Equivalent of the PostGIS ST_BuildArea() function.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.build_area(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def make_valid(self):
        """
        Repair invalid geometries.

        Two methods are available:

        the 'linework' algorithm tries to preserve every edge and vertex in the input. It combines all rings into a set of noded lines and then extracts valid polygons from that linework. An alternating even-odd strategy is used to assign areas as interior or exterior. A disadvantage is that for some relatively simple invalid geometries this produces rather complex results.
        the 'structure' algorithm tries to reason from the structure of the input to find the 'correct' repair: exterior rings bound area, interior holes exclude area. It first makes all rings valid, then shells are merged and holes are subtracted from the shells to generate valid result. It assumes that holes and shells are correctly categorized in the input geometry.
        TODO check input parameters for this function
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.make_valid(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def normalize(self):
        """
        Convert Geometry to strict normal form (or canonical form).

        In strict canonical form <canonical-form>, the coordinates, rings of a polygon and parts of multi geometries are ordered consistently. Typically useful for testing purposes (for example in combination with equals_exact).
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.normalize(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def node(self):
        """
        Return the fully noded version of the linear input as MultiLineString.

        Given a linear input geometry, this function returns a new MultiLineString in which no lines cross each other but only touch at and points. To obtain this, all intersections between segments are computed and added to the segments, and duplicate segments are removed.

        Non-linear input (points) will result in an empty MultiLineString.

        This function can for example be used to create a fully-noded linework suitable to passed as input to polygonize.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.node(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def point_on_surface(self):
        """
        Return a point that intersects an input geometry.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.point_on_surface(s_arr))
        return SpatialSeries._to_spatialseries([result], crs_wkt)

    def polygonize(self):
        """
        Create polygons formed from the linework of a set of Geometries.

        Polygonizes an array of Geometries that contain linework which represents the edges of a planar graph. Any type of Geometry may be provided as input; only the constituent lines and rings will be used to create the output polygons.

        Lines or rings that when combined do not completely close a polygon will result in an empty GeometryCollection. Duplicate segments are ignored.

        This function returns the polygons within a GeometryCollection. Individual Polygons can be obtained using get_geometry to get a single polygon or get_parts to get an array of polygons. MultiPolygons can be constructed from the output using shapely.multipolygons(shapely.get_parts(shapely.polygonize(geometries))).
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.polygonize(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def remove_repeated_points(self, tolerance=0.0):
        """
        Return a copy of a Geometry with repeated points removed.

        From the start of the coordinate sequence, each next point within the tolerance is removed.

        Removing repeated points with a non-zero tolerance may result in an invalid geometry being returned.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.remove_repeated_points(s_arr, tolerance=tolerance)
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def reverse(self):
        """
        Return a copy of a Geometry with the order of coordinates reversed.

        If a Geometry is a polygon with interior rings, the interior rings are also reversed.

        Points are unchanged. None is returned where Geometry is None.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.reverse(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def simplify(self, tolerance=0.0, preserve_topology=True):
        """
        Return a simplified version of an input geometry.

        The Douglas-Peucker algorithm is used to simplify the geometry.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.simplify(
                s_arr, tolerance=tolerance, preserve_topology=preserve_topology
            )
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def snap(self, reference, tolerance=0.0):
        """
        Snap the vertices and segments of the geometry to vertices of the reference.

        Vertices and segments of the input geometry are snapped to vertices of the reference geometry, returning a new geometry; the input geometries are not modified. The result geometry is the input geometry with the vertices and segments snapped. If no snapping occurs then the input geometry is returned unchanged. The tolerance is used to control where snapping is performed.

        Where possible, this operation tries to avoid creating invalid geometries; however, it does not guarantee that output geometries will be valid. It is the responsibility of the caller to check for and handle invalid geometries.

        Because too much snapping can result in invalid geometries being created, heuristics are used to determine the number and location of snapped vertices that are likely safe to snap. These heuristics may omit some potential snaps that are otherwise within the tolerance.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.snap(s_arr, reference=reference, tolerance=tolerance)
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def oriented_envelope(self):
        """
        Compute the oriented envelope (minimum rotated rectangle) of the input geometry.

        The oriented envelope encloses an input geometry, such that the resulting rectangle has minimum area.

        Unlike envelope this rectangle is not constrained to be parallel to the coordinate axes. If the convex hull of the object is a degenerate (line or point) this degenerate is returned.

        The starting point of the rectangle is not fixed. You can use ~shapely.normalize to reorganize the rectangle to strict canonical form <canonical-form> so the starting point is always the lower left point.

        minimum_rotated_rectangle is an alias for oriented_envelope.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.oriented_envelope(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def minimum_rotated_rectangle(self):
        """
        Compute the oriented envelope (minimum rotated rectangle) of the input geometry.

        The oriented envelope encloses an input geometry, such that the resulting rectangle has minimum area.

        Unlike envelope this rectangle is not constrained to be parallel to the coordinate axes. If the convex hull of the object is a degenerate (line or point) this degenerate is returned.

        The starting point of the rectangle is not fixed. You can use ~shapely.normalize to reorganize the rectangle to strict canonical form <canonical-form> so the starting point is always the lower left point.

        minimum_rotated_rectangle is an alias for oriented_envelope.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.minimum_rotated_rectangle(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def minimum_bounding_circle(self):
        """
        Compute the minimum bounding circle that encloses an input geometry.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.minimum_bounding_circle(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    # Linestring operations
    def line_interpolate_point(self, distance, normalized=False):
        """
        Return a point interpolated at given distance on a line.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(
            shapely.line_interpolate_point(
                s_arr, distance=distance, normalized=normalized
            )
        )
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def line_locate_point(self, other, normalized=False):
        """
        Return the distance to the line origin of given point.

        If given point does not intersect with the line, the point will first be projected onto the line after which the distance is taken.
        """
        s_arr = self.to_shapely_array()
        result = shapely.line_locate_point(s_arr, other, normalized=normalized)
        return result

    def line_merge(self, directed=False):
        """
        Return (Multi)LineStrings formed by combining the lines in a MultiLineString.

        Lines are joined together at their endpoints in case two lines are intersecting. Lines are not joined when 3 or more lines are intersecting at the endpoints. Line elements that cannot be joined are kept as is in the resulting MultiLineString.

        The direction of each merged LineString will be that of the majority of the LineStrings from which it was derived. Except if directed=True is specified, then the operation will not change the order of points within lines and so only lines which can be joined with no change in direction are merged.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.line_merge(s_arr, directed=directed))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def shortest_line(self, other):
        """
        Return the shortest line between two geometries.

        The resulting line consists of two points, representing the nearest points between the geometry pair. The line always starts in the first geometry a and ends in the second geometry b. The endpoints of the line will not necessarily be existing vertices of the input geometries a and b, but can also be a point along a line segment.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.shortest_line(s_arr, other))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    # geometry properties
    def force_2d(self):
        """
        Force the dimensionality of a geometry to 2D.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.force_2d(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def force_3d(self):
        """
        Force the dimensionality of a geometry to 3D.

        2D geometries will get the provided Z coordinate; Z coordinates of 3D geometries are unchanged (unless they are nan).

        Note that for empty geometries, 3D is only supported since GEOS 3.9 and then still only for simple geometries (non-collections).
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.force_3d(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def get_coordinate_dimension(self):
        """
        Return the dimensionality of the coordinates in a geometry (2, 3 or 4).

        The return value can be one of the following:

        Return 2 for geometries with XY coordinate types,

        Return 3 for XYZ or XYM coordinate types (distinguished by has_z() or has_m()),

        Return 4 for XYZM coordinate types,

        Return -1 for missing geometries (None values).

        Note that with GEOS < 3.12, if the first Z coordinate equals nan, this function will return 2. Geometries with M coordinates are supported with GEOS >= 3.12.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_coordinate_dimension(s_arr)
        return result

    def get_dimensions(self):
        """
        Return the inherent dimensionality of a geometry.

        The inherent dimension is 0 for points, 1 for linestrings and linearrings, and 2 for polygons. For geometrycollections it is the max of the containing elements. Empty collections and None values return -1.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_dimensions(s_arr)
        return result

    def get_exterior_ring(self):
        """
        Return the exterior ring of a polygon.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.get_exterior_ring(s_arr))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def get_geometry(self, index: int):
        """
        Return the nth geometry from a collection of geometries.

        Parameters
        ----------
        index
            Negative values count from the end of the collection backwards.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.get_geometry(s_arr, index))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def get_interior_ring(self, index: int):
        """
        Return the nth interior ring of a polygon.

        The number of interior rings in non-polygons equals zero.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.get_interior_ring(s_arr, index))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def get_num_coordinates(self):
        """
        Return the total number of coordinates in a geometry.

        Returns 0 for not-a-geometry values.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_num_coordinates(s_arr)
        return result

    def get_num_interior_rings(self):
        """
        Return number of internal rings in a polygon.

        Returns 0 for not-a-geometry values.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_num_interior_rings(s_arr)
        return result

    def get_num_points(self):
        """
        Return the number of points in a linestring or linearring.

        Returns 0 for not-a-geometry values. The number of points in geometries other than linestring or linearring equals zero.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_num_points(s_arr)
        return result

    def get_point(self, index: int):
        """
        Return the nth point of a linestring or linearring.
        """
        s_arr = self.to_shapely_array()
        crs_wkt = self._s.spatial.get_crs()
        result = shapely.to_wkb(shapely.get_point(s_arr, index))
        return SpatialSeries._to_spatialseries(result, crs_wkt)

    def get_type_id(self):
        """
        Return the type ID of a geometry.

        Possible values are:

        None (missing) is -1

        POINT is 0

        LINESTRING is 1

        LINEARRING is 2

        POLYGON is 3

        MULTIPOINT is 4

        MULTILINESTRING is 5

        MULTIPOLYGON is 6

        GEOMETRYCOLLECTION is 7
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_type_id(s_arr)
        return result

    def get_x(self):
        """
        Return the x-coordinate of a point.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_x(s_arr)
        return result

    def get_y(self):
        """
        Return the y-coordinate of a point.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_y(s_arr)
        return result

    def get_z(self):
        """
        Return the z-coordinate of a point.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_z(s_arr)
        return result

    def get_m(self):
        """
        Return the m-coordinate of a point.
        """
        s_arr = self.to_shapely_array()
        result = shapely.get_m(s_arr)
        return result
