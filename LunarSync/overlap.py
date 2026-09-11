"""
Stage 2 of the pipeline.

The goal here is simple: before we start doing expensive feature matching,
figure out which part of the reference image actually covers the source
image's footprint.

This is especially important for lunar data because a NAC/WAC image can be
much larger than the small region we are interested in. Matching against the
entire reference image wastes both computation and memory, and can also
produce false matches from completely unrelated terrain.

The basic approach is:

    source corners
          ↓
    latitude / longitude
          ↓
    reference pixel coordinates
          ↓
    bounding box + safety margin
          ↓
    reference ROI

The pixel <-> lat/lon relationship is approximated from the four image
corners. This is a coarse geographic prior, not a full sensor model. The
actual geometric registration is handled later by feature matching and
RANSAC.

The Haversine distance uses the Moon's radius rather than Earth's radius.
That distinction matters when working with lunar coordinates.
"""

import numpy as np

import LunarSync.config as config


def haversine_km(lat1, lon1, lat2, lon2, radius_km=config.MOON_RADIUS_KM):
    """Return the great-circle distance between two lunar coordinates."""

    # Convert everything to radians before applying the Haversine formula.
    # Using the Moon's radius keeps the result physically meaningful for
    # lunar coordinates instead of accidentally treating them as Earth data.
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius_km * c


def _corner_points(meta):
    """
    Return the four image-corner correspondences.

    Each entry contains:

        (row, col, latitude, longitude)

    The corners are always returned in clockwise image order:

        upper-left → upper-right → lower-right → lower-left

    Keeping this order consistent is important because these points are used
    to build the approximate image-to-geographic relationship later.
    """

    corners = meta["corners"]

    # Without geographic corner coordinates there is no reliable way to
    # estimate where this image sits inside the reference image.
    if corners is None:
        raise ValueError(
            "This source's metadata has no corner coordinates -- "
            "cannot compute an overlap window without them."
        )

    n_lines, n_samples = meta["shape"]

    # Pixel coordinates are zero-based, so the last valid row/column is
    # dimension - 1 rather than dimension itself.
    last_row, last_col = n_lines - 1, n_samples - 1

    return [
        (
            0,
            0,
            corners["upper_left_latitude"],
            corners["upper_left_longitude"],
        ),
        (
            0,
            last_col,
            corners["upper_right_latitude"],
            corners["upper_right_longitude"],
        ),
        (
            last_row,
            last_col,
            corners["lower_right_latitude"],
            corners["lower_right_longitude"],
        ),
        (
            last_row,
            0,
            corners["lower_left_latitude"],
            corners["lower_left_longitude"],
        ),
    ]


def fit_latlon_to_pixel(meta):
    """
    Build an approximate (latitude, longitude) -> (row, col) mapping.

    We only know the geographic coordinates of the four image corners, so
    we fit a simple affine relationship using those points.

    This is deliberately a coarse model. It is good enough to answer:

        "Where should this part of the source roughly appear
         inside the reference image?"

    It is NOT intended to replace a proper spacecraft/sensor geometry model.
    """

    points = _corner_points(meta)

    rows = np.array([p[0] for p in points], dtype=np.float64)
    cols = np.array([p[1] for p in points], dtype=np.float64)
    lats = np.array([p[2] for p in points], dtype=np.float64)
    lons = np.array([p[3] for p in points], dtype=np.float64)

    # We model both image coordinates as a linear function of latitude,
    # longitude, and a constant offset:
    #
    #     row = a*lat + b*lon + c
    #     col = d*lat + e*lon + f
    #
    # Four corner points give us enough equations to estimate these
    # coefficients with least squares.
    A = np.column_stack([lats, lons, np.ones_like(lats)])

    row_coeffs, _, _, _ = np.linalg.lstsq(A, rows, rcond=None)
    col_coeffs, _, _, _ = np.linalg.lstsq(A, cols, rcond=None)

    def latlon_to_pixel(lat, lon):
        """
        Convert geographic coordinates into approximate image coordinates.

        Keeping this as a small function makes it convenient to project many
        geographic points without rebuilding the model every time.
        """

        row = (
            row_coeffs[0] * lat
            + row_coeffs[1] * lon
            + row_coeffs[2]
        )

        col = (
            col_coeffs[0] * lat
            + col_coeffs[1] * lon
            + col_coeffs[2]
        )

        return row, col

    return latlon_to_pixel


def compute_overlap_window(source_meta, reference_meta, margin_px=200):
    """
    Estimate the part of the reference image that overlaps the source.

    The source's four geographic corners are projected into reference-image
    pixel coordinates. Their bounding box becomes the initial ROI.

    A small margin is added because the corner-based mapping is only an
    approximation. The extra pixels give the later matching stage some room
    to handle residual geometric error.

    Returns:
        {
            "row0": ...,
            "row1": ...,
            "col0": ...,
            "col1": ...
        }

    All coordinates are expressed in the reference image's pixel space.
    """

    # First build the approximate geographic -> reference-pixel mapping.
    # We only need the reference model here because the source corners are
    # already expressed in latitude/longitude.
    ref_to_pixel = fit_latlon_to_pixel(reference_meta)

    source_corners = _corner_points(source_meta)

    ref_rows = []
    ref_cols = []

    # Ask the reference model where each source corner should land.
    # Four projected corners are enough to form a conservative rectangular ROI.
    for _, _, lat, lon in source_corners:
        row, col = ref_to_pixel(lat, lon)
        ref_rows.append(row)
        ref_cols.append(col)

    ref_n_lines, ref_n_samples = reference_meta["shape"]

    # Expand the estimated footprint slightly. This protects us from small
    # errors in the metadata-derived mapping and gives feature matching some
    # breathing room around the predicted overlap.
    row0 = max(
        int(np.floor(min(ref_rows))) - margin_px,
        0,
    )

    row1 = min(
        int(np.ceil(max(ref_rows))) + margin_px,
        ref_n_lines,
    )

    col0 = max(
        int(np.floor(min(ref_cols))) - margin_px,
        0,
    )

    col1 = min(
        int(np.ceil(max(ref_cols))) + margin_px,
        ref_n_samples,
    )

    # If this happens, either the footprints do not overlap or something is
    # wrong with the geographic metadata. Failing here is much better than
    # passing an invalid crop to the matcher and debugging the failure later.
    if row1 <= row0 or col1 <= col0:
        raise ValueError(
            "Computed overlap window is empty or inverted -- the source and reference "
            "footprints may not actually overlap, or one metadata dict has bad corners."
        )

    return {
        "row0": row0,
        "row1": row1,
        "col0": col0,
        "col1": col1,
    }


def crop_to_overlap(reference_array, window):
    """
    Extract the previously calculated reference ROI.

    This works with both normal NumPy arrays and memory-mapped arrays, which
    means we can crop a very large reference image without first loading the
    entire image into RAM.
    """

    return np.asarray(
        reference_array[
            window["row0"]:window["row1"],
            window["col0"]:window["col1"],
        ]
    )


def align_shapes(source_array, reference_array):
    """
    Trim both images to a common height and width.

    This is only a bookkeeping step for the tiling/matching stage. It should
    NOT be confused with actual image registration.

    After resampling, the two images may still have slightly different
    dimensions. Since the tiled matcher expects comparable array sizes,
    we keep the largest common region.

    Any real geometric mismatch is still handled later by feature matching,
    spatial filtering, and the final RANSAC-based transformation.

    Call this after resampling and immediately before matching.
    """

    common_h = min(
        source_array.shape[0],
        reference_array.shape[0],
    )

    common_w = min(
        source_array.shape[1],
        reference_array.shape[1],
    )

    return (
        source_array[:common_h, :common_w],
        reference_array[:common_h, :common_w],
    )


def corner_bbox_latlon(meta):
    """
    Return the latitude/longitude bounding box of an image footprint.

    This is mainly a convenience function for sanity checks and plotting.
    It is kept separate from compute_overlap_window because sometimes we only
    want to inspect the geographic extent without actually projecting it into
    reference-image coordinates.
    """

    points = _corner_points(meta)

    lats = [p[2] for p in points]
    lons = [p[3] for p in points]

    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
    }