import polars as pl
import shapely

from .io import spatial_series_dtype


class GeometryProperties:
    """Expressions derived from shapely's [geometry properties](https://shapely.readthedocs.io/en/stable/properties.html)"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def force_2d(self):
        """
        Force the dimensionality of a geometry to 2D.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.force_2d(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def force_3d(self):
        """
        Force the dimensionality of a geometry to 3D.

        2D geometries will get the provided Z coordinate; Z coordinates of 3D geometries are unchanged (unless they are nan).

        Note that for empty geometries, 3D is only supported since GEOS 3.9 and then still only for simple geometries (non-collections).
        """
        return self._expr.map_batches(
            lambda s: s.spatial.force_3d(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

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
        return self._expr.map_batches(
            lambda s: s.spatial.get_coordinate_dimension(),
            return_dtype=pl.Int8,
            is_elementwise=True,
        )

    def get_dimensions(self):
        """
        Return the inherent dimensionality of a geometry.

        The inherent dimension is 0 for points, 1 for linestrings and linearrings, and 2 for polygons. For geometrycollections it is the max of the containing elements. Empty collections and None values return -1.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_dimensions(),
            return_dtype=pl.Int8,
            is_elementwise=True,
        )

    def get_exterior_ring(self):
        """
        Return the exterior ring of a polygon.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_exterior_ring(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def get_geometry(self, index: int):
        """
        Return the nth geometry from a collection of geometries.

        Parameters
        ----------
        index
            Negative values count from the end of the collection backwards.
        """
        if index is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.get_geometry(index),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.get_geometry(
                    combined.struct[1],
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def get_interior_ring(self, index: int):
        """
        Return the nth interior ring of a polygon.

        The number of interior rings in non-polygons equals zero.
        """
        if index is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.get_interior_ring(index),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.get_interior_ring(
                    combined.struct[1],
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def get_num_coordinates(self):
        """
        Return the total number of coordinates in a geometry.

        Returns 0 for not-a-geometry values.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_num_coordinates(),
            return_dtype=pl.Int32,
            is_elementwise=True,
        )

    def get_num_interior_rings(self):
        """
        Return number of internal rings in a polygon.

        Returns 0 for not-a-geometry values.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_num_interior_rings(),
            return_dtype=pl.Int32,
            is_elementwise=True,
        )

    def get_num_points(self):
        """
        Return the number of points in a linestring or linearring.

        Returns 0 for not-a-geometry values. The number of points in geometries other than linestring or linearring equals zero.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_num_points(),
            return_dtype=pl.Int32,
            is_elementwise=True,
        )

    def get_point(self, index: int):
        """
        Return the nth point of a linestring or linearring.
        """
        if index is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.get_point(index),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.get_point(
                    combined.struct[1],
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

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
        return self._expr.map_batches(
            lambda s: s.spatial.get_type_id(),
            return_dtype=pl.Int8,
            is_elementwise=True,
        )

    def get_x(self):
        """
        Return the x-coordinate of a point.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_x(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )

    def get_y(self):
        """
        Return the y-coordinate of a point.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_y(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )

    def get_z(self):
        """
        Return the z-coordinate of a point.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_z(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )

    def get_m(self):
        """
        Return the m-coordinate of a point.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.get_m(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )


class Measurement:
    """Expressions derived from shapely's [Measurements](https://shapely.readthedocs.io/en/stable/measurement.html)"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def area(self):
        """
        Compute the area of a (multi)polygon.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.area(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )

    def distance(self, other=None):
        """
        Compute the Cartesian distance between two geometries.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry) for details.
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.distance(other),
                return_dtype=pl.Float64,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.distance(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Float64,
                is_elementwise=True,
            )

    def bounds(self):
        """
        Compute the bounds (extent) of a geometry.

        For each geometry these 4 numbers are returned as a struct: min x, min y, max x, max y.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.bounds(),
            return_dtype=pl.Array(pl.Float64, 4),
            is_elementwise=True,
        )

    def length(self):
        """
        Compute the length of a (multi)linestring or polygon perimeter.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.length(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )

    def hausdorff_distance(self, other=None, densify=None):
        """
        Compute the discrete Hausdorff distance between two geometries.

        The Hausdorff distance is a measure of similarity: it is the greatest distance between any point in A and the closest point in B. The discrete distance is an approximation of this metric: only vertices are considered. The parameter `densify` makes this approximation less coarse by splitting the line segments between vertices before computing the distance.

        Parameters
        ----------
        other
            A shapely geometry object

        densify
            The value of densify is required to be between 0 and 1.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry) for details.
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.hausdorff_distance(other, densify),
                return_dtype=pl.Float64,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.hausdorff_distance(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                    densify=densify,
                ),
                return_dtype=pl.Float64,
                is_elementwise=True,
            )

    def frechet_distance(self, other=None, densify=None):
        """
        Compute the discrete Fréchet distance between two geometries.

        The Fréchet distance is a measure of similarity: it is the greatest distance between any point in A and the closest point in B. The discrete distance is an approximation of this metric: only vertices are considered. The parameter `densify` makes this approximation less coarse by splitting the line segments between vertices before computing the distance.

        Fréchet distance sweep continuously along their respective curves and the direction of curves is significant. This makes it a better measure of similarity than Hausdorff distance for curve or surface matching.

        Parameters
        ----------
        other
            A shapely geometry object

        densify
            The value of densify is required to be between 0 and 1.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry) for details.
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.frechet_distance(other, densify),
                return_dtype=pl.Float64,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.frechet_distance(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                    densify=densify,
                ),
                return_dtype=pl.Float64,
                is_elementwise=True,
            )

    def minimum_clearance(self):
        """
        Compute the Minimum Clearance distance.

        A geometry's "minimum clearance" is the smallest distance by which a vertex of the geometry could be moved to produce an invalid geometry.

        If no minimum clearance exists for a geometry (for example, a single point, or an empty geometry), infinity is returned.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.minimum_clearance(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )

    def minimum_bounding_radius(self):
        """
        Compute the radius of the minimum bounding circle of an input geometry.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.minimum_bounding_radius(),
            return_dtype=pl.Float64,
            is_elementwise=True,
        )


class Predicates:
    """Expressions derived from shapely's [Predicates](https://shapely.readthedocs.io/en/stable/predicates.html)"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def has_z(self):
        """
        Return True if a geometry has Z coordinates.

        Note that for GEOS < 3.12 this function returns False if the (first) Z coordinate equals NaN.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.has_z(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_ccw(self):
        """
        Return True if a linestring or linearring is counterclockwise.

        Note that there are no checks on whether lines are actually closed and not self-intersecting, while this is a requirement for is_ccw. The recommended usage of this function for linestrings is is_ccw(g) & is_simple(g) and for linearrings is_ccw(g) & is_valid(g).
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_ccw(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_closed(self):
        """
        Return True if a linestring's first and last points are equal.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_closed(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_empty(self):
        """
        Return True if a geometry is an empty point, polygon, etc.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_empty(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_geometry(self):
        """
        Return True if the object is a geometry.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_geometry(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_missing(self):
        """
        Return True if the object is not a geometry (None).
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_missing(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_ring(self):
        """
        Return True if a linestring is closed and simple.

        This function will return False for non-linestrings.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_ring(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_simple(self):
        """
        Return True if the geometry is simple.

        A simple geometry has no anomalous geometric points, such as self-intersections or self tangency.

        Note that polygons and linearrings are assumed to be simple. Use is_valid to check these kind of geometries for self-intersections.

        This function will return False for geometrycollections.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_simple(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_valid(self):
        """
        Return True if a geometry is well formed.

        Returns False for missing values.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_valid(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_valid_input(self):
        """
        Return True if the object is a geometry or None.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_valid_input(),
            return_dtype=pl.Boolean,
            is_elementwise=True,
        )

    def is_valid_reason(self):
        """
        Return a string stating if a geometry is valid and if not, why.

        Returns None for missing values.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.is_valid_reason(),
            return_dtype=pl.String,
            is_elementwise=True,
        )

    def crosses(self, other=None):
        """
        Return True if A and B spatially cross.

        A crosses B if they have some but not all interior points in common, the intersection is one dimension less than the maximum dimension of A or B, and the intersection is not equal to either A or B.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.crosses(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.crosses(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def contains(self, other=None):
        """
        Return True if geometry B is completely inside geometry A.

        A contains B if no points of B lie in the exterior of A and at least one point of the interior of B lies in the interior of A.

        Parameters
        ----------
        other
            A shapely geometry object

        Note
        ----
        Following this definition, a geometry does not contain its boundary, but it does contain itself. See contains_properly for a version where a geometry does not contain itself.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.contains(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.contains(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def contains_properly(self, other=None):
        """
        Return True if geometry B is completely inside geometry A, with no common boundary points.

        A contains B properly if B intersects the interior of A but not the boundary (or exterior). This means that a geometry A does not "contain properly" itself, which contrasts with the contains function, where common points on the boundary are allowed.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.contains_properly(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.contains_properly(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def covered_by(self, other=None):
        """
        Return True if no point in geometry A is outside geometry B.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.covered_by(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.covered_by(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def covers(self, other=None):
        """
        Return True if no point in geometry B is outside geometry A.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.covers(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.covers(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def disjoint(self, other=None):
        """
        Return True if A and B do not share any point in space.

        Disjoint implies that overlaps, touches, within, and intersects are False. Note missing (None) values are never disjoint.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.disjoint(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.disjoint(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def equals(self, other=None):
        """
        Return True if A and B are spatially equal.

        If A is within B and B is within A, A and B are considered equal. The ordering of points can be different.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.equals(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.equals(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def intersects(self, other=None):
        """
        Return True if A and B share any portion of space.

        Intersects implies that overlaps, touches, covers, or within are True.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.intersects(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.intersects(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def overlaps(self, other=None):
        """
        Return True if A and B spatially overlap.

        A and B overlap if they have some but not all points/space in common, have the same dimension, and the intersection of the interiors of the two geometries has the same dimension as the geometries themselves. That is, only polyons can overlap other polygons and only lines can overlap other lines. If A covers or is within B, overlaps won't be True.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.overlaps(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.overlaps(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def touches(self, other=None):
        """
        Return True if the only points shared between A and B are on their boundaries.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.touches(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.touches(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def within(self, other=None):
        """
        Return True if geometry A is completely inside geometry B.

        A is within B if no points of A lie in the exterior of B and at least one point of the interior of A lies in the interior of B.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.within(other),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.within(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def relate(self, other=None):
        """
        Return a string representation of the DE-9IM intersection matrix.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.relate(other),
                return_dtype=pl.String,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: shapely.relate(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=pl.String,
                is_elementwise=True,
            )

    def contains_xy(self, x=None, y=None):
        """
        Return True if the Point (x, y) is completely inside geom.

        This is a special-case (and faster) variant of the contains function which avoids having to create a Point object if you start from x/y coordinates.

        Note that in the case of points, the contains_properly predicate is equivalent to contains.

        See the docstring of contains for more details about the predicate.

        Parameters
        ----------
        x
            The X coordinate to check

        y
            The Y coordinate to check

        One geometry different x/y coordinate input
        -------------------------------------------
        **To compute between the values in the series and a single x,y pair** provide the `x` and `y` parameters.

        **To compute between two geometries in a column and columns of x/y coordinates of the frame** wrap all columns into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)

        """
        if x is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.contains_xy(x, y),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with a geometry and x/y fields.
            return self._expr.map_batches(
                lambda combined: shapely.contains_xy(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1],
                    combined.struct[2],
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def dwithin(self, other=None, distance=None):
        """
        Return True if the geometries are within a given distance.

        Using this function is more efficient than computing the distance and comparing the result.

        Parameters
        ----------
        other
            A shapely geometry object

        distance
            The distance to check if the geometries are within

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.dwithin(other, distance),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with a two geometry and distance fields.
            return self._expr.map_batches(
                lambda combined: shapely.dwithin(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                    combined.struct[2],
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def intersects_xy(self, x, y):
        """
        Return True if geom and the Point (x, y) share any portion of space.

        This is a special-case (and faster) variant of the intersects function which avoids having to create a Point object if you start from x/y coordinates.

        See the docstring of intersects for more details about the predicate.

        Parameters
        ----------
        x
            The X coordinate to check

        y
            The Y coordinate to check

        One geometry different x/y coordinate input
        -------------------------------------------
        **To compute between the values in the series and a single x,y pair** provide the `x` and `y` parameters.

        **To compute between two geometries in a column and columns of x/y coordinates of the frame** wrap all columns into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if x is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.intersects_xy(x, y),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with a geometry and x/y fields.
            return self._expr.map_batches(
                lambda combined: shapely.intersects_xy(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1],
                    combined.struct[2],
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def equals_exact(self, other, tolerance):
        """
        Return True if the geometries are structurally equivalent within a given tolerance.

        This method uses exact coordinate equality, which requires coordinates to be equal (within specified tolerance) and in the same order for all components (vertices, rings, or parts) of a geometry. This is in contrast with the equals function which uses spatial (topological) equality and does not require all components to be in the same order. Because of this, it is possible for equals to be True while equals_exact is False.

        The order of the coordinates can be normalized (by setting the normalize keyword to True) so that this function will return True when geometries are structurally equivalent but differ only in the ordering of vertices. However, this function will still return False if the order of interior rings within a Polygon or the order of geometries within a multi geometry are different.

        Parameters
        ----------
        other
            A shapely geometry object

        tolerance
            The tolerance to use in the comparison.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.equals_exact(other, tolerance),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometry and tolerance fields.
            return self._expr.map_batches(
                lambda combined: shapely.equals_exact(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                    combined.struct[2],
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )

    def relate_pattern(self, other, pattern):
        """
        Return True if the DE-9IM relationship code satisfies the pattern.

        This function compares the DE-9IM code string for two geometries against a specified pattern. If the string matches the pattern then True is returned, otherwise False. The pattern specified can be an exact match (0, 1 or 2), a boolean match (uppercase T or F), or a wildcard (*). For example, the pattern for the within predicate is 'T*F**F***'.

        Parameters
        ----------
        other
            A shapely geometry object

        pattern
            The pattern to match the DE-9IM relationship code against.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.relate_pattern(other, pattern),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometry and pattern fields.
            return self._expr.map_batches(
                lambda combined: shapely.relate_pattern(
                    combined.struct[0].spatial.to_shapely_array(),
                    combined.struct[1].spatial.to_shapely_array(),
                    combined.struct[2],
                ),
                return_dtype=pl.Boolean,
                is_elementwise=True,
            )


class SetOperations:
    """Expressions derived from shapely's [Set Operations](https://shapely.readthedocs.io/en/stable/set_operations.html)"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def difference(self, other=None, grid_size=None):
        """
        Return the part of geometry A that does not intersect with geometry B.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.

        Parameters
        ----------
        other
            A shapely geometry object

        grid_size
            Precision grid size; will use the highest precision of the inputs by default.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.difference(other, grid_size),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.difference(
                    combined.struct[1].spatial.to_shapely_array(),
                    grid_size,
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def intersection(self, other=None, grid_size=None):
        """
        Return the geometry that is shared between input geometries.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.

        Parameters
        ----------
        other
            A shapely geometry object

        grid_size
            Precision grid size; will use the highest precision of the inputs by default.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.intersection(other, grid_size),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.intersection(
                    combined.struct[1].spatial.to_shapely_array(),
                    grid_size,
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def intersection_all(self, grid_size=None):
        """
        Return the intersection of multiple geometries.

        This function ignores None values when other Geometry elements are present. If all elements of the given axis are None, an empty GeometryCollection is returned.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.

        Parameters
        ----------
        grid_size
            Precision grid size; will use the highest precision of the inputs by default.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.intersection_all(grid_size),
            return_dtype=spatial_series_dtype,
            returns_scalar=True,
        )

    def symmetric_difference(self, other=None, grid_size=None):
        """
        Return the geometry with the portions of input geometries that do not intersect.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.

        Parameters
        ----------
        other
            A shapely geometry object

        grid_size
            Precision grid size; will use the highest precision of the inputs by default.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.symmetric_difference(other, grid_size),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.symmetric_difference(
                    combined.struct[1].spatial.to_shapely_array(), grid_size
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def union(self, other=None, grid_size=None):
        """
        Merge geometries into one.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.

        Parameters
        ----------
        other
            A shapely geometry object

        grid_size
            Precision grid size; will use the highest precision of the inputs by default.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.union(other, grid_size),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.union(
                    combined.struct[1].spatial.to_shapely_array(), grid_size
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def union_all(self, grid_size=None):
        """
        Return the union of multiple geometries.

        This function ignores None values when other Geometry elements are present. If all elements of the given axis are None an empty GeometryCollection is returned.

        If grid_size is nonzero, input coordinates will be snapped to a precision grid of that size and resulting coordinates will be snapped to that same grid. If 0, this operation will use double precision coordinates. If None, the highest precision of the inputs will be used, which may be previously set using set_precision. Note: returned geometry does not have precision set unless specified previously by set_precision.

        Parameters
        ----------
        grid_size
            Precision grid size; will use the highest precision of the inputs by default.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.union_all(grid_size),
            return_dtype=spatial_series_dtype,
            returns_scalar=True,
        )


class ConstructiveOperations:
    """Expressions derived from shapely's [Constructive Operations](https://shapely.readthedocs.io/en/stable/constructive.html)"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def boundary(self):
        """
        Return the topological boundary of a geometry.

        This function will return None for geometrycollections.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.boundary(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def buffer(
        self,
        distance,
        quad_segs=8,
        cap_style="round",
        join_style="round",
        mitre_limit=5.0,
        single_sided=False,
    ):
        """
        Compute the buffer of a geometry for positive and negative buffer distance.

        The buffer of a geometry is defined as the Minkowski sum (or difference, for negative distance) of the geometry with a circle with radius equal to the absolute value of the buffer distance.

        The buffer operation always returns a polygonal result. The negative or zero-distance buffer of lines and points is always empty.

        Parameters
        ----------
        distance
            Specifies the circle radius in the Minkowski sum (or difference).

        quad_segs
            Specifies the number of linear segments in a quarter circle in the approximation of circular arcs.

        cap_style
            Specifies the shape of buffered line endings. BufferCapStyle.round (`round`) results in circular line endings (see quad_segs). Both BufferCapStyle.square (`square`) and BufferCapStyle.flat (`flat`) result in rectangular line endings, only BufferCapStyle.flat (`flat`) will end at the original vertex, while BufferCapStyle.square (`square`) involves adding the buffer width.

        join_style
            Specifies the shape of buffered line midpoints. BufferJoinStyle.round (`round`) results in rounded shapes. BufferJoinStyle.bevel (`bevel`) results in a beveled edge that touches the original vertex. BufferJoinStyle.mitre (`mitre`) results in a single vertex that is beveled depending on the mitre_limit parameter.

        mitre_limit
            Crops of `mitre`-style joins if the point is displaced from the buffered vertex by more than this limit.

        single_sided
            Only buffer at one side of the geometry.

        """
        return self._expr.map_batches(
            lambda s: s.spatial.buffer(
                distance,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                mitre_limit=mitre_limit,
                single_sided=single_sided,
            ),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def offset_curve(self, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
        """
        Return a (Multi)LineString at a distance from the object.

        For positive distance the offset will be at the left side of the input line. For a negative distance it will be at the right side. In general, this function tries to preserve the direction of the input.

        Note: the behaviour regarding orientation of the resulting line depends on the GEOS version. With GEOS < 3.11, the line retains the same direction for a left offset (positive distance) or has opposite direction for a right offset (negative distance), and this behaviour was documented as such in previous Shapely versions. Starting with GEOS 3.11, the function tries to preserve the orientation of the original line.

        Parameters
        ----------
        distance
            Specifies the circle radius in the Minkowski sum (or difference).

        quad_segs
            Specifies the number of linear segments in a quarter circle in the approximation of circular arcs.

        join_style
            Specifies the shape of buffered line midpoints. BufferJoinStyle.round (`round`) results in rounded shapes. BufferJoinStyle.bevel (`bevel`) results in a beveled edge that touches the original vertex. BufferJoinStyle.mitre (`mitre`) results in a single vertex that is beveled depending on the mitre_limit parameter.

        mitre_limit
            Crops of `mitre`-style joins if the point is displaced from the buffered vertex by more than this limit.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.offset_curve(
                distance,
                quad_segs=quad_segs,
                join_style=join_style,
                mitre_limit=mitre_limit,
            ),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def centroid(self):
        """
        Compute the geometric center (center-of-mass) of a geometry.

        For multipoints this is computed as the mean of the input coordinates. For multilinestrings the centroid is weighted by the length of each line segment. For multipolygons the centroid is weighted by the area of each polygon.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.centroid(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def clip_by_rect(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """
        Return the portion of a geometry within a rectangle.

        The geometry is clipped in a fast but possibly dirty way. The output is not guaranteed to be valid. No exceptions will be raised for topological errors.

        Note: empty geometries or geometries that do not overlap with the specified bounds will result in GEOMETRYCOLLECTION EMPTY.

        Parameters
        ----------
        xmin
            Minimum x value of the rectangle.

        ymin
            Minimum y value of the rectangle.

        xmax
            Maximum x value of the rectangle.

        ymax
            Maximum y value of the rectangle.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.clip_by_rect(
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
            ),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def concave_hull(self, ratio=0.0, allow_holes=False):
        """
        Compute a concave geometry that encloses an input geometry.

        Parameters
        ----------
        ratio
            Number in the range [0, 1]. Higher numbers will include fewer vertices in the hull.

        allow_holes
            If set to True, the concave hull may have holes.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.concave_hull(ratio=ratio, allow_holes=allow_holes),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def convex_hull(self):
        """
        Compute the minimum convex geometry that encloses an input geometry.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.convex_hull(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def delaunay_triangles(self, tolerance=0.0, only_edges=False):
        """
        Compute a Delaunay triangulation around the vertices of an input geometry.

        The output is a geometrycollection containing polygons (default) or linestrings (see only_edges). Returns an empty geometry for input geometries that contain less than 3 vertices.

        Parameters
        ----------
        tolerance
            Snap input vertices together if their distance is less than this value.

        only_edges
            If set to True, the triangulation will return a collection of linestrings instead of polygons.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.delaunay_triangles(
                tolerance=tolerance,
                only_edges=only_edges,
            ),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def segmentize(self, max_segment_length: float):
        """
        Add vertices to line segments based on maximum segment length.

        Additional vertices will be added to every line segment in an input geometry so that segments are no longer than the provided maximum segment length. New vertices will evenly subdivide each segment.

        Only linear components of input geometries are densified; other geometries are returned unmodified.

        Parameters
        ----------
        max_segment_length
            Additional vertices will be added so that all line segments are no longer than this value. Must be greater than 0.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.segmentize(max_segment_length=max_segment_length),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def envelope(self):
        """
        Compute the minimum bounding box that encloses an input geometry.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.envelope(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def extract_unique_points(self):
        """
        Return all distinct vertices of an input geometry as a multipoint.

        Note that only 2 dimensions of the vertices are considered when testing for equality.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.extract_unique_points(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def build_area(self):
        """
        Create an areal geometry formed by the constituent linework of given geometry.

        Equivalent of the PostGIS ST_BuildArea() function.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.build_area(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def make_valid(self):
        """
        Repair invalid geometries.

        Two methods are available:

        the 'linework' algorithm tries to preserve every edge and vertex in the input. It combines all rings into a set of noded lines and then extracts valid polygons from that linework. An alternating even-odd strategy is used to assign areas as interior or exterior. A disadvantage is that for some relatively simple invalid geometries this produces rather complex results.
        the 'structure' algorithm tries to reason from the structure of the input to find the 'correct' repair: exterior rings bound area, interior holes exclude area. It first makes all rings valid, then shells are merged and holes are subtracted from the shells to generate valid result. It assumes that holes and shells are correctly categorized in the input geometry.
        TODO check input parameters for this function
        """
        return self._expr.map_batches(
            lambda s: s.spatial.make_valid(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def normalize(self):
        """
        Convert Geometry to strict normal form (or canonical form).

        In strict canonical form <canonical-form>, the coordinates, rings of a polygon and parts of multi geometries are ordered consistently. Typically useful for testing purposes (for example in combination with equals_exact).
        """
        return self._expr.map_batches(
            lambda s: s.spatial.normalize(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def node(self):
        """
        Return the fully noded version of the linear input as MultiLineString.

        Given a linear input geometry, this function returns a new MultiLineString in which no lines cross each other but only touch at and points. To obtain this, all intersections between segments are computed and added to the segments, and duplicate segments are removed.

        Non-linear input (points) will result in an empty MultiLineString.

        This function can for example be used to create a fully-noded linework suitable to passed as input to polygonize.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.node(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def point_on_surface(self):
        """
        Return a point that intersects an input geometry.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.point_on_surface(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def polygonize(self):
        """
        Create polygons formed from the linework of a set of Geometries.

        Polygonizes an array of Geometries that contain linework which represents the edges of a planar graph. Any type of Geometry may be provided as input; only the constituent lines and rings will be used to create the output polygons.

        Lines or rings that when combined do not completely close a polygon will result in an empty GeometryCollection. Duplicate segments are ignored.

        This function returns the polygons within a GeometryCollection. Individual Polygons can be obtained using get_geometry to get a single polygon or get_parts to get an array of polygons. MultiPolygons can be constructed from the output using shapely.multipolygons(shapely.get_parts(shapely.polygonize(geometries))).
        """
        return self._expr.map_batches(
            lambda s: s.spatial.polygonize(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def remove_repeated_points(self, tolerance=0.0):
        """
        Return a copy of a Geometry with repeated points removed.

        From the start of the coordinate sequence, each next point within the tolerance is removed.

        Removing repeated points with a non-zero tolerance may result in an invalid geometry being returned.

        Parameters
        ----------
        tolerance
            Use 0.0 to remove only exactly repeated points.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.remove_repeated_points(tolerance=tolerance),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def reverse(self):
        """
        Return a copy of a Geometry with the order of coordinates reversed.

        If a Geometry is a polygon with interior rings, the interior rings are also reversed.

        Points are unchanged. None is returned where Geometry is None.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.reverse(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def simplify(self, tolerance=0.0, preserve_topology=True):
        """
        Return a simplified version of an input geometry.

        The Douglas-Peucker algorithm is used to simplify the geometry.

        Parameters
        ----------
        tolerance
            The maximum allowed geometry displacement. The higher this value, the smaller the number of vertices in the resulting geometry.

        preserve_topology
            By default (True), the operation will avoid creating invalid geometries (checking for collapses, ring-intersections, etc), but this is computationally more expensive.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.simplify(
                tolerance=tolerance, preserve_topology=preserve_topology
            ),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def snap(self, reference=None, tolerance: float = None):
        """
        Snap the vertices and segments of the geometry to vertices of the reference.

        Vertices and segments of the input geometry are snapped to vertices of the reference geometry, returning a new geometry; the input geometries are not modified. The result geometry is the input geometry with the vertices and segments snapped. If no snapping occurs then the input geometry is returned unchanged. The tolerance is used to control where snapping is performed.

        Where possible, this operation tries to avoid creating invalid geometries; however, it does not guarantee that output geometries will be valid. It is the responsibility of the caller to check for and handle invalid geometries.

        Because too much snapping can result in invalid geometries being created, heuristics are used to determine the number and location of snapped vertices that are likely safe to snap. These heuristics may omit some potential snaps that are otherwise within the tolerance.

        Parameters
        ----------
        reference
            Geometry or geometries to snap to.

        tolerance
            The maximum distance between the input and reference geometries for snapping to occur. A value of 0 will snap only identical points.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `reference` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)
        """
        if reference is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.difference(reference, tolerance),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.difference(
                    combined.struct[1].spatial.to_shapely_array(),
                    tolerance,
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def oriented_envelope(self):
        """
        Compute the oriented envelope (minimum rotated rectangle) of the input geometry.

        The oriented envelope encloses an input geometry, such that the resulting rectangle has minimum area.

        Unlike envelope this rectangle is not constrained to be parallel to the coordinate axes. If the convex hull of the object is a degenerate (line or point) this degenerate is returned.

        The starting point of the rectangle is not fixed. You can use ~shapely.normalize to reorganize the rectangle to strict canonical form <canonical-form> so the starting point is always the lower left point.

        minimum_rotated_rectangle is an alias for oriented_envelope.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.oriented_envelope(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def minimum_rotated_rectangle(self):
        """
        Compute the oriented envelope (minimum rotated rectangle) of the input geometry.

        The oriented envelope encloses an input geometry, such that the resulting rectangle has minimum area.

        Unlike envelope this rectangle is not constrained to be parallel to the coordinate axes. If the convex hull of the object is a degenerate (line or point) this degenerate is returned.

        The starting point of the rectangle is not fixed. You can use ~shapely.normalize to reorganize the rectangle to strict canonical form <canonical-form> so the starting point is always the lower left point.

        minimum_rotated_rectangle is an alias for oriented_envelope.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.minimum_rotated_rectangle(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def minimum_bounding_circle(self):
        """
        Compute the minimum bounding circle that encloses an input geometry.
        """
        return self._expr.map_batches(
            lambda s: s.spatial.minimum_bounding_circle(),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )


class LinestringOperations:
    """Expressions derived from shapely's [Constructive Operations](https://shapely.readthedocs.io/en/stable/linear.html)"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def line_interpolate_point(self, distance=None, normalized=False):
        """
        Return a point interpolated at given distance on a line.

        Parameters
        ----------
        distance
            Negative values measure distance from the end of the line. Out-of-range values will be clipped to the line endings.

        normalized
            If True, the distance is a fraction of the total line length instead of the absolute distance.

        Note
        ----
        **To compute between the values in the series and a scalar distance** provide the distance to the `distance` parameter.

        **To compute between the geometries and a column in the frame for the distance** wrap the geometry and other column of distances into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)

        """
        if distance is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.line_interpolate_point(
                    distance=distance, normalized=normalized
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct geometry and number.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.line_interpolate_point(
                    combined.struct[1],
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def line_locate_point(self, other, normalized=False):
        """
        Return the distance to the line origin of given point.

        If given point does not intersect with the line, the point will first be projected onto the line after which the distance is taken.

        Parameters
        ----------
        other
            Point or points to calculate the distance from.

        normalized
             If True, the distance is a fraction of the total line length instead of the absolute distance.

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)

        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.line_locate_point(other, normalized=normalized),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.line_locate_point(
                    combined.struct[1].spatial.to_shapely_array(),
                    normalized=normalized,
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )

    def line_merge(self, directed=False):
        """
        Return (Multi)LineStrings formed by combining the lines in a MultiLineString.

        Lines are joined together at their endpoints in case two lines are intersecting. Lines are not joined when 3 or more lines are intersecting at the endpoints. Line elements that cannot be joined are kept as is in the resulting MultiLineString.

        The direction of each merged LineString will be that of the majority of the LineStrings from which it was derived. Except if directed=True is specified, then the operation will not change the order of points within lines and so only lines which can be joined with no change in direction are merged.

        Parameters
        ----------
        directed
            Only combine lines if possible without changing point order. Requires GEOS >= 3.11.0
        """
        return self._expr.map_batches(
            lambda s: s.spatial.line_merge(directed=directed),
            return_dtype=spatial_series_dtype,
            is_elementwise=True,
        )

    def shortest_line(self, other):
        """
        Return the shortest line between two geometries.

        The resulting line consists of two points, representing the nearest points between the geometry pair. The line always starts in the first geometry a and ends in the second geometry b. The endpoints of the line will not necessarily be existing vertices of the input geometries a and b, but can also be a point along a line segment.

        Parameters
        ----------
        other
            A shapely geometry object

        Two geometry input
        ------------------
        **To compute between the values in the series and a scalar geometry** provide the other geometry to the `other` parameter.

        **To compute between two geometries in different columns of the frame** wrap both geometries into a struct before using the expression.
        See [Spatial expressions which use more than geometry](index.md#spatial-expressions-which-use-more-than-geometry)

        """
        if other is not None:
            return self._expr.map_batches(
                lambda s: s.spatial.shortest_line(other),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )
        else:
            # expect struct with two geometries.
            return self._expr.map_batches(
                lambda combined: combined.struct[0].spatial.shortest_line(
                    combined.struct[1].spatial.to_shapely_array(),
                ),
                return_dtype=spatial_series_dtype,
                is_elementwise=True,
            )


@pl.api.register_expr_namespace("spatial")
class SpatialExpr(
    GeometryProperties,
    Measurement,
    Predicates,
    SetOperations,
    ConstructiveOperations,
    LinestringOperations,
):
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def to_shapely_array(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda s: s.spatial.to_shapely_array(),
            return_dtype=pl.Object,
            is_elementwise=True,
        )

    def reproject(self, crs_to):
        """
        Reproject data to a different CRS.

        Parameters
        ----------
        crs_to
            The coordinate reference system to reproject the data into.
        """
        # using is_elementwise=True causes issues,
        # assuming something with the LRU caching or something like that.
        return self._expr.map_batches(
            lambda s: s.spatial.reproject(crs_to),
            return_dtype=spatial_series_dtype,
        )

    def min_max(self):
        """
        Normalizes a value in a column to be 0-1
        """
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())
