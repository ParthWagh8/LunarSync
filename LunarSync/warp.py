"""
Stage 8:
    Warp the source image into the reference image's pixel frame using
    the estimated geometric transformation.

Stage 9:
    Record where the warped image sits within the reference coordinate
    system.

The warp is destination-tiled rather than source-tiled. In other words,
we divide the output/reference image into manageable tiles and work out
which part of the source image is actually needed for each tile.

This is important for large lunar images such as OHRC, where loading the
entire image and performing one large warp could consume a lot of RAM.
"""
import numpy as np
import cv2

import LunarSync.config as config


def tiled_warp(source_array, reference_shape, H,
                margin=100, max_dim=config.WARP_MAX_DIM, overlap=config.WARP_OVERLAP):
    """
    Warp the source image into a canvas having the same size as the
    reference image.

    Instead of warping the complete image in one operation, the reference
    canvas is processed tile-by-tile. For every destination tile, we use
    the inverse homography to determine which region of the source image
    contributes to that tile.

    This keeps memory usage roughly bounded by the tile size, even when
    working with very large lunar images.
    """
    h, w = reference_shape
    # Convert to float64 before doing matrix operations. Homographies can
    # become numerically unstable if we rely on integer or low-precision
    # representations here.
    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("Invalid homography passed to tiled_warp.")
    
    # OpenCV maps source -> destination, but for a destination tile we need
    # to answer the opposite question: "Which source pixels end up here?"
    # Hence we work with the inverse transformation.

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        raise ValueError("Homography is singular -- check n_inliers/inlier_ratio, the fit is degenerate.")

    # The final image has the same dimensions as the reference frame.
    # 'filled' keeps track of pixels that have already been written,
    # which becomes important when tiles overlap.
    warped = np.zeros((h, w), dtype=np.uint8)
    filled = np.zeros((h, w), dtype=bool)

    # Leave a small overlap between neighbouring tiles. This helps avoid
    # tiny gaps at tile boundaries caused by rounding during coordinate
    # conversion.
    step = max(256, int(max_dim) - int(overlap))

    for dy0 in range(0, h, step):
        dy1 = min(dy0 + max_dim, h)

        for dx0 in range(0, w, step):
            dx1 = min(dx0 + max_dim, w)

            # Four corners are enough to describe the destination tile.
            # We map these corners backwards into the source image to find
            # the smallest source region that could contribute to the tile.
            dest_corners = np.array([[dx0, dy0], [dx1 - 1, dy0],
                                      [dx1 - 1, dy1 - 1], [dx0, dy1 - 1]],
                                     dtype=np.float32).reshape(-1, 1, 2)
            src_corners = cv2.perspectiveTransform(dest_corners, H_inv).reshape(-1, 2)

            # Add a margin around the estimated source region. The extra
            # pixels give interpolation some breathing room near the edges
            # and make the tile slightly more robust to rounding.
            sx0 = max(int(np.floor(src_corners[:, 0].min())) - margin, 0)
            sy0 = max(int(np.floor(src_corners[:, 1].min())) - margin, 0)
            sx1 = min(int(np.ceil(src_corners[:, 0].max())) + margin + 1, source_array.shape[1])
            sy1 = min(int(np.ceil(src_corners[:, 1].max())) + margin + 1, source_array.shape[0])

            # The tile may fall completely outside the source image.
            # There is nothing to warp in that case.
            if sx1 <= sx0 or sy1 <= sy0:
                continue

            # Only copy the source region required for this destination
            # tile. This is the main reason the tiled approach saves RAM.
            source_tile = np.ascontiguousarray(source_array[sy0:sy1, sx0:sx1])

            # The source tile has its own local origin at (0, 0), while
            # the homography was estimated in full-image coordinates.
            # Translate the source coordinates so the transformation still
            # refers to the correct pixels.
            translate_to_source = np.array([[1, 0, sx0], [0, 1, sy0], [0, 0, 1]], dtype=np.float64)

            # OpenCV will create the destination tile starting at (0, 0),
            # so shift the full reference coordinates back to the tile's
            # local coordinate system.            
            translate_to_dest_local = np.array([[1, 0, -dx0], [0, 1, -dy0], [0, 0, 1]], dtype=np.float64)

            # Combine the coordinate translations with the original
            # source -> reference homography.
            local_H = translate_to_dest_local @ H @ translate_to_source

            dest_w, dest_h = dx1 - dx0, dy1 - dy0

            # Warp only this small source region. Linear interpolation is
            # a reasonable choice here because the final product is an
            # image registration output rather than a nearest-neighbour
            # label/mask image.
            tile_warp = cv2.warpPerspective(source_tile, local_H, (dest_w, dest_h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            dest_region = warped[dy0:dy1, dx0:dx1]
            dest_filled = filled[dy0:dy1, dx0:dx1]

            # If neighbouring tiles overlap, keep the first valid value.
            # This prevents later tiles from unnecessarily overwriting
            # pixels that have already been populated.
            write_mask = (tile_warp > 0) & (~dest_filled)

            dest_region[write_mask] = tile_warp[write_mask]
            dest_filled[write_mask] = True

    return warped


def piecewise_warp(source_array, reference_shape, segments,
                    margin=100, max_dim=config.WARP_MAX_DIM, overlap=config.WARP_OVERLAP):
    """
    Warp the source using multiple transformations, one for each
    along-track segment.

    A single homography may not describe the complete image accurately
    when the imaging geometry changes across a long lunar strip. In that
    situation, the fitted piecewise model provides a separate local
    transformation for each segment.

    Segment overlaps are resolved using a simple "first valid pixel wins"
    strategy.
    """
    h, w = reference_shape
    warped = np.zeros((h, w), dtype=np.uint8)
    filled = np.zeros((h, w), dtype=bool)

    for segment in segments:

        # Restrict processing to the portion of the source image covered
        # by this segment. The margin provides a small buffer around the
        # fitted segment boundaries.
        y0 = max(int(segment["y_min"]) - margin, 0)
        y1 = min(int(segment["y_max"]) + margin, source_array.shape[0])

        if y1 <= y0:
            continue

        segment_source = source_array[y0:y1, :]

        # After cropping the source, its row 0 is no longer the original
        # image's row 0. Translate it back into the original coordinate
        # system before applying the fitted segment homography.
        row_shift = np.array([[1, 0, 0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
        segment_H = segment["H"] @ row_shift

        # Reuse the same memory-efficient tiled warp used for the global
        # homography case.
        segment_warp = tiled_warp(segment_source, (h, w), segment_H,
                                   margin=margin, max_dim=max_dim, overlap=overlap)
        
        # Combine this segment with whatever has already been produced.
        # We deliberately do not overwrite pixels filled by an earlier
        # segment.        
        write_mask = (segment_warp > 0) & (~filled)
        warped[write_mask] = segment_warp[write_mask]
        filled |= write_mask

    return warped


def warp_source(source_array, reference_shape, fit_result,
                 margin=100, max_dim=config.WARP_MAX_DIM, overlap=config.WARP_OVERLAP):
    """
    Apply the appropriate warp based on the fitted geometry model.

    The geometry-fitting stage can return either a single global
    homography or a piecewise model. Keeping that decision here means
    the rest of the pipeline does not need to know how the transformation
    was fitted.
    """
    if fit_result["mode"] == "piecewise":
        return piecewise_warp(source_array, reference_shape, fit_result["segments"],
                               margin=margin, max_dim=max_dim, overlap=overlap)
    return tiled_warp(source_array, reference_shape, fit_result["H"],
                       margin=margin, max_dim=max_dim, overlap=overlap)


def reanchor_to_reference(reference_meta, overlap_window):
    """
    Stage 9: Coordinate-system integration.

    This function records how the warped image relates to the reference
    image's coordinate system.

    It does not create a true GeoTIFF or perform a full CRS/geodetic
    transformation. Instead, it stores the reference origin, crop offset,
    and pixel resolution so the spatial relationship is preserved in the
    accompanying metadata.
    """
    corners = reference_meta.get("corners")

    # Without reference corner coordinates we cannot describe where the
    # warped image sits geographically.
    if corners is None:
        return {"note": "reference metadata has no corners -- cannot anchor coordinates"}

    return {
        "reference_top_left_lat": corners["upper_left_latitude"],
        "reference_top_left_lon": corners["upper_left_longitude"],
        "crop_row0": overlap_window["row0"],
        "crop_col0": overlap_window["col0"],
        "reference_pixel_resolution_m": reference_meta.get("pixel_resolution_m"),
        "note": ("Warped product's pixel (0,0) corresponds to the reference's full-frame "
                 "pixel (crop_row0, crop_col0), which in turn sits at "
                 "(reference_top_left_lat, reference_top_left_lon) plus "
                 "(crop_row0, crop_col0) * reference_pixel_resolution_m along the reference's "
                 "own axes. This is bookkeeping, not a full CRS-aware GeoTIFF transform."),
    }
