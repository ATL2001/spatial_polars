#![allow(clippy::unused_unit)]
use byteorder::{ByteOrder, LittleEndian};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn get_geom_base_type_and_bytes_per_point(wkb: &[u8], byte_offset: usize) -> (i32, usize) {
    // Ensure we can safely read the geometry type
    if wkb.len() < byte_offset + 5 {
        return (0, 0);
    }

    let geom_type_int = LittleEndian::read_i32(&wkb[byte_offset + 1..byte_offset + 5]);
    let geom_coordinate_thousands = geom_type_int / 1000;
    let geom_base_type = geom_type_int % 1000;

    let geom_coordinate_dimensions_num: usize = match geom_coordinate_thousands {
        0 => 2,             // XY
        1 => 3,             // XYZ
        2 => 3,             // XYM
        3 => 4,             // XYZM
        _ => return (0, 0), // Handle unexpected dimension codes safely
    };

    // return the number of dimensions * 8 bytes for each f64
    let bytes_per_point: usize = geom_coordinate_dimensions_num * 8;
    (geom_base_type, bytes_per_point)
}

pub fn compute_polygon_area(coordinates: &[(f64, f64)]) -> f64 {
    // If there are fewer than 3 coordinates, a polygon cannot be formed.
    if coordinates.len() < 3 {
        return 0.0;
    }

    let mut shoelace = 0.0;
    for window in coordinates.windows(2) {
        let cur = window[0];
        let nxt = window[1];
        shoelace += cur.0 * nxt.1;
        shoelace -= cur.1 * nxt.0;
    }
    (shoelace / 2.0).abs()
}

fn get_polygon_ring_coordinates(
    wkb: &[u8],
    start_of_ring: usize,
    bytes_per_point: usize,
) -> Option<Vec<(f64, f64)>> {
    if wkb.len() < start_of_ring + 4 {
        return None;
    }

    let points_in_ring_int =
        LittleEndian::read_i32(&wkb[start_of_ring..start_of_ring + 4]) as usize;
    let mut ring_coordinates = Vec::with_capacity(points_in_ring_int);

    for point_num in 0..points_in_ring_int {
        let p_start = start_of_ring + 4 + point_num * bytes_per_point;

        let x_start = p_start;
        let x_end = p_start + 8;
        let y_start = p_start + 8;
        let y_end = p_start + 16;

        // Explicit bounds check to prevent panic on truncated binary data
        if wkb.len() < y_end {
            return None;
        }

        let point_x_f = LittleEndian::read_f64(&wkb[x_start..x_end]);
        let point_y_f = LittleEndian::read_f64(&wkb[y_start..y_end]);

        ring_coordinates.push((point_x_f, point_y_f));
    }
    Some(ring_coordinates)
}

fn get_area_of_polygon(wkb: &[u8]) -> Option<f64> {
    // Get the area from the wkb of a single polygon
    let (geom_base_type, bytes_per_point) = get_geom_base_type_and_bytes_per_point(wkb, 0);
    if bytes_per_point == 0 || (geom_base_type != 3 && geom_base_type != 6) {
        return None;
    }

    let mut polygon_area = 0.0;
    let num_rings_int = LittleEndian::read_i32(&wkb[5..9]) as usize;
    let mut start_of_ring = 9; // skip the first 9 bytes (byte order, wkb type, number of rings)

    for ring_num in 0..num_rings_int {
        // Check boundary before parsing the number of points in this ring
        if wkb.len() < start_of_ring + 4 {
            return None;
        }
        let points_in_ring_int =
            LittleEndian::read_i32(&wkb[start_of_ring..start_of_ring + 4]) as usize;

        let ring_coordinates = get_polygon_ring_coordinates(&wkb, start_of_ring, bytes_per_point)?;
        let ring_area = compute_polygon_area(&ring_coordinates);

        if ring_num == 0 {
            polygon_area += ring_area;
        } else {
            polygon_area -= ring_area;
        }

        start_of_ring = start_of_ring + 4 + points_in_ring_int * bytes_per_point;
    }
    Some(polygon_area)
}

pub fn get_area(wkb: &[u8], byte_offset: usize) -> Option<f64> {
    // Ensure we can safely read the geometry type (indices 1 to 5)
    if wkb.len() < 5 {
        return None;
    }
    let (geom_base_type, _bytes_per_point) =
        get_geom_base_type_and_bytes_per_point(wkb, byte_offset);
    let mut polygon_area = 0.0;

    if geom_base_type == 3 {
        //polygon
        if wkb.len() < 9 {
            return None;
        }
        polygon_area = get_area_of_polygon(wkb)?;
    } else if geom_base_type == 6 {
        //multipolygon
        if wkb.len() < 9 {
            return None;
        }
        let polygons_wkb = get_polygons_from_multipolygon(wkb).unwrap_or(vec![]);
        for polygon_wkb in &polygons_wkb {
            polygon_area += get_area_of_polygon(polygon_wkb)?;
        }
    } else if geom_base_type == 7 {
        //geometrycollection
        if wkb.len() < 9 {
            return None;
        }
        let geometries_wkb = get_geometries_from_geometry_collection(wkb).unwrap_or(vec![]);

        for geometry_wkb in &geometries_wkb {
            polygon_area += get_area(geometry_wkb, 0)?;
        }
    }
    return Some(polygon_area);
}

#[polars_expr(output_type=Float64)]
fn area(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca: &BinaryChunked = s.binary()?;
    let out: Float64Chunked = ca
        .into_iter()
        .map(|opt_bytes| match opt_bytes {
            Some(bytes) => get_area(bytes, 0),
            None => Some(f64::NAN),
        })
        .collect();
    Ok(out.into_series())
}

fn get_next_line_start_from_multi(wkb: &[u8], start_of_line: usize) -> Option<usize> {
    let (_geom_base_type, bytes_per_point) =
        get_geom_base_type_and_bytes_per_point(wkb, start_of_line);

    let mut start_of_next_line = start_of_line + 9;
    let num_lines_int = LittleEndian::read_i32(&wkb[start_of_line + 5..start_of_line + 9]) as usize;

    for _line_num in 0..num_lines_int {
        // Check boundary before parsing the number of points in this line
        if wkb.len() < start_of_next_line + 4 {
            return None;
        }
        let points_in_line_int =
            LittleEndian::read_i32(&wkb[start_of_next_line..start_of_next_line + 4]) as usize;

        start_of_next_line = start_of_next_line + 4 + points_in_line_int * bytes_per_point;
    }
    return Some(start_of_next_line);
}

fn get_next_polygon_start_from_multi(wkb: &[u8], start_of_polygon: usize) -> Option<usize> {
    let (_geom_base_type, bytes_per_point) =
        get_geom_base_type_and_bytes_per_point(wkb, start_of_polygon);

    let mut start_of_next_ring = start_of_polygon + 9;
    let num_rings_int =
        LittleEndian::read_i32(&wkb[start_of_polygon + 5..start_of_polygon + 9]) as usize;

    for _ring_num in 0..num_rings_int {
        // Check boundary before parsing the number of points in this ring
        if wkb.len() < start_of_next_ring + 4 {
            return None;
        }
        let points_in_ring_int =
            LittleEndian::read_i32(&wkb[start_of_next_ring..start_of_next_ring + 4]) as usize;

        start_of_next_ring = start_of_next_ring + 4 + points_in_ring_int * bytes_per_point;
    }
    return Some(start_of_next_ring);
}

pub fn get_polygons_from_multipolygon(wkb: &[u8]) -> Option<Vec<&[u8]>> {
    if wkb.len() < 9 {
        // Ensure we can safely read the geometry type
        return None;
    }
    let (whole_geom_base_type, _whole_geom_bytes_per_point) =
        get_geom_base_type_and_bytes_per_point(wkb, 0);

    if whole_geom_base_type != 6 {
        // should have been a multipolygon, return None
        return None;
    }
    let num_polygons_int = LittleEndian::read_i32(&wkb[5..9]) as usize;
    let mut start_of_next_polygon: usize = 9;
    let mut polygons: Vec<&[u8]> = vec![];
    for _polygon_num in 0..num_polygons_int {
        let start_of_this_polygon = start_of_next_polygon;
        start_of_next_polygon = get_next_polygon_start_from_multi(wkb, start_of_next_polygon)?;
        let this_polygon = &wkb[start_of_this_polygon..start_of_next_polygon];
        polygons.push(this_polygon);
    }
    return Some(polygons);
}

fn get_next_geometry_start_from_collection(wkb: &[u8], start_of_geometry: usize) -> Option<usize> {
    let (geom_base_type, bytes_per_point) =
        get_geom_base_type_and_bytes_per_point(wkb, start_of_geometry);

    if geom_base_type == 1 {
        // point
        return Some(start_of_geometry + 5 + bytes_per_point);
    } else if geom_base_type == 2 {
        // linestring
        let num_points =
            LittleEndian::read_i32(&wkb[start_of_geometry + 5..start_of_geometry + 9]) as usize;
        return Some(start_of_geometry + 9 + bytes_per_point * num_points);
    } else if geom_base_type == 3 {
        // polygon
        let num_rings_int =
            LittleEndian::read_i32(&wkb[start_of_geometry + 5..start_of_geometry + 9]) as usize;
        let mut start_of_next_ring = start_of_geometry + 9; // skip the first 9 bytes (byte order, wkb type, number of rings)
        for _ring_num in 0..num_rings_int {
            // Check boundary before parsing the number of points in this ring
            if wkb.len() < start_of_next_ring + 4 {
                return None;
            }
            let points_in_ring_int =
                LittleEndian::read_i32(&wkb[start_of_next_ring..start_of_next_ring + 4]) as usize;
            start_of_next_ring = start_of_next_ring + 4 + points_in_ring_int * bytes_per_point;
        }
        return Some(start_of_next_ring);
    } else if geom_base_type == 4 {
        // multipoint
        let num_points_int =
            LittleEndian::read_i32(&wkb[start_of_geometry + 5..start_of_geometry + 9]) as usize;
        let mut start_of_next_point = start_of_geometry + 9; // skip the first 9 bytes (byte order, wkb type, number of points)
        for _point_num in 0..num_points_int {
            // Check boundary before parsing the number of points in this ring
            if wkb.len() < start_of_next_point + 5 {
                return None;
            }
            let (_this_point_base_type, this_point_bytes_per_point) =
                get_geom_base_type_and_bytes_per_point(wkb, start_of_next_point);

            start_of_next_point = start_of_next_point + 5 + this_point_bytes_per_point;
        }
        return Some(start_of_next_point);
    } else if geom_base_type == 5 {
        // multilinestring
        let num_lines_int =
            LittleEndian::read_i32(&wkb[start_of_geometry + 5..start_of_geometry + 9]) as usize;
        let mut start_of_next_line: usize = start_of_geometry + 9;
        for _polygon_num in 0..num_lines_int {
            start_of_next_line = get_next_line_start_from_multi(wkb, start_of_next_line)?;
        }
        return Some(start_of_next_line);
    } else if geom_base_type == 6 {
        // multipolygon
        let num_polygons_int =
            LittleEndian::read_i32(&wkb[start_of_geometry + 5..start_of_geometry + 9]) as usize;
        let mut start_of_next_polygon: usize = start_of_geometry + 9;
        for _polygon_num in 0..num_polygons_int {
            start_of_next_polygon = get_next_polygon_start_from_multi(wkb, start_of_next_polygon)?;
        }
        return Some(start_of_next_polygon);
    } else if geom_base_type == 7 {
        // geometrycollection
        return Some(get_next_geometry_start_from_collection(
            &wkb,
            start_of_geometry + 9,
        )?);
    } else {
        return None;
    }
}

pub fn get_geometries_from_geometry_collection(wkb: &[u8]) -> Option<Vec<&[u8]>> {
    if wkb.len() < 9 {
        // Ensure we can safely read the geometry type
        return None;
    }
    let (whole_geom_base_type, _whole_geom_bytes_per_point) =
        get_geom_base_type_and_bytes_per_point(wkb, 0);

    if whole_geom_base_type != 7 {
        // should have been a GeometryCollection, return None
        return None;
    }
    let num_geometries = LittleEndian::read_i32(&wkb[5..9]) as usize;
    let mut start_of_next_geometry: usize = 9;
    let mut geometries: Vec<&[u8]> = vec![];
    for _geometry_num in 0..num_geometries {
        let start_of_this_geometry = start_of_next_geometry;
        start_of_next_geometry =
            get_next_geometry_start_from_collection(wkb, start_of_this_geometry)?;
        let this_geometry = &wkb[start_of_this_geometry..start_of_next_geometry];

        geometries.push(this_geometry);
    }
    return Some(geometries);
}
