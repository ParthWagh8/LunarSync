"""
Stage 7 of the pipeline: turning raw SuperGlue matches into a trusted
geometric transform.

RANSAC is used to find the homography supported by the largest consistent
group of matches while rejecting incorrect correspondences. A quality gate
then checks whether enough reliable matches remain for the transformation
to be considered trustworthy.

The module also supports a piecewise fitting mode. Instead of forcing one
homography over the entire image, the source image can be divided into
along-track bands and a separate homography can be fitted to each band.

This is useful for pushbroom sensors such as OHRC, TMC-2 and IIRS, where
the viewing geometry can change gradually along a long scan strip. In
those cases, a single global homography may work well in one part of the
image but accumulate error toward the other end.
"""

import cv2
import numpy as np

import LunarSync.config as config


def fit_homography(
    mkpts_src,
    mkpts_ref,
    ransac_thresh=config.RANSAC_REPROJ_THRESH_PX
):
    """
    Fit a single 3x3 homography using RANSAC.

    If this OpenCV build provides USAC_MAGSAC, it is preferred because it
    gives a more robust estimate in the presence of noisy correspondences.
    Otherwise, the standard OpenCV RANSAC implementation is used.

    Returns:
        H             : 3x3 homography matrix, or None if fitting fails
        inlier_mask   : boolean mask identifying geometrically consistent
                        matches
        n_inliers     : number of inlier correspondences
        inlier_ratio  : fraction of all matches classified as inliers

    Homography estimation needs at least four point correspondences. Rather
    than raising an exception for smaller inputs, the function returns a
    failed-fit result so the caller can decide how to handle it.
    """

    # Four point pairs are the minimum required to estimate a projective
    # transformation. This check also prevents OpenCV from being called
    # with an obviously insufficient set of correspondences.
    if len(mkpts_src) < 4:
        return None, None, 0, 0.0

    # Prefer MAGSAC++ when the installed OpenCV version supports it.
    # The fallback keeps the pipeline compatible with older OpenCV builds.
    ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

    H, mask = cv2.findHomography(
        mkpts_src,
        mkpts_ref,
        method=ransac_method,
        ransacReprojThreshold=ransac_thresh
    )

    # OpenCV can fail to estimate a transformation even when enough points
    # are supplied, for example when the points do not define a valid
    # projective configuration.
    if H is None:
        return None, None, 0, 0.0

    # OpenCV returns the inlier mask as an integer array. Convert it to a
    # boolean mask so it can be used directly for indexing the match arrays.
    inlier_mask = mask.ravel().astype(bool)

    n_inliers = int(inlier_mask.sum())
    inlier_ratio = n_inliers / len(mkpts_src)

    return H, inlier_mask, n_inliers, inlier_ratio


def fit_piecewise(
    mkpts_src,
    mkpts_ref,
    n_segments=config.PIECEWISE_N_SEGMENTS,
    ransac_thresh=config.RANSAC_REPROJ_THRESH_PX,
    min_pts_per_segment=config.PIECEWISE_MIN_PTS_PER_SEGMENT,
):
    """
    Divide the source matches into along-track bands and fit one homography
    per band.

    A segment is skipped when it does not contain enough matches to support
    a reliable homography. This is preferable to forcing a transformation
    from a very small number of correspondences.

    Returns:
        segments:
            List of dictionaries describing the successful segment fits.

        inlier_mask:
            One boolean mask covering the original match array. A match is
            marked True only when it was an inlier in the homography fitted
            to its own segment.

    Each segment dictionary contains:
        {
            "y_min": ...,
            "y_max": ...,
            "H": ...,
            "n_inliers": ...,
            "inlier_ratio": ...
        }
    """

    # The source y-coordinate corresponds to the along-track direction.
    # Dividing this range into bands lets the transformation vary gradually
    # instead of assuming one homography is valid over the entire strip.
    y_min, y_max = mkpts_src[:, 1].min(), mkpts_src[:, 1].max()
    edges = np.linspace(y_min, y_max, n_segments + 1)

    segments = []

    # Keep the mask in the coordinate system of the original match arrays.
    # This makes it possible for later stages to recover the actual points
    # that survived all piecewise fits.
    inlier_mask = np.zeros(len(mkpts_src), dtype=bool)

    for i in range(n_segments):
        lo, hi = edges[i], edges[i + 1]

        # Select only matches whose source points fall inside this
        # along-track band.
        in_band = (
            (mkpts_src[:, 1] >= lo)
            & (mkpts_src[:, 1] <= hi)
        )

        # Sparse bands do not provide enough geometric information for a
        # dependable fit, so simply leave them out.
        if in_band.sum() < min_pts_per_segment:
            continue

        H, band_inliers, n_inliers, inlier_ratio = fit_homography(
            mkpts_src[in_band],
            mkpts_ref[in_band],
            ransac_thresh
        )

        if H is None:
            continue

        # fit_homography returned indices relative to this band. Convert
        # them back to indices in the original match arrays before updating
        # the global inlier mask.
        band_indices = np.where(in_band)[0]
        inlier_mask[band_indices[band_inliers]] = True

        segments.append({
            "y_min": float(lo),
            "y_max": float(hi),
            "H": H,
            "n_inliers": n_inliers,
            "inlier_ratio": inlier_ratio
        })

    # If every segment failed, there is no useful geometric model to return.
    # This is more informative than silently returning an empty result.
    if not segments:
        raise ValueError(
            "No segment had enough matches to fit a homography -- "
            "reduce n_segments, or check the raw match count from matching.py."
        )

    return segments, inlier_mask


def uniformity_coverage(
    mkpts_ref,
    ref_shape,
    grid=config.UNIFORM_GRID
):
    """
    Measure how evenly the reference-image inliers are distributed.

    The reference image is divided into a grid, and the function calculates
    the fraction of cells containing at least one inlier point.

    This function only MEASURES coverage. It does not remove or redistribute
    points. Actual enforcement of the uniform-distribution requirement is
    performed by uniform_select() below.

    Returns:
        float in [0, 1]
    """

    if len(mkpts_ref) == 0:
        return 0.0

    gh, gw = int(grid[0]), int(grid[1])
    h, w = int(ref_shape[0]), int(ref_shape[1])

    # Convert pixel coordinates into grid-cell coordinates.
    # Clipping protects against points that lie exactly on the image border
    # and would otherwise produce an index equal to the grid size.
    gx = np.clip(
        (mkpts_ref[:, 0] / max(w, 1) * gw).astype(int),
        0,
        gw - 1
    )
    gy = np.clip(
        (mkpts_ref[:, 1] / max(h, 1) * gh).astype(int),
        0,
        gh - 1
    )

    # A set removes duplicate cell locations, leaving only the cells that
    # actually contain at least one correspondence.
    filled = set(zip(gx.tolist(), gy.tolist()))

    return len(filled) / float(gh * gw)


def median_reproj_error(
    mkpts_src,
    mkpts_ref,
    H,
    inlier_mask
):
    """
    Calculate the median Euclidean reprojection error over inlier matches.

    The median is useful alongside RMSE because a few unusually large errors
    can strongly influence RMSE. Lunar imagery can contain repeated or
    visually ambiguous terrain, making a robust error statistic valuable.

    Returns None when there is no valid homography or no inliers.
    """

    # Import locally to keep the function self-contained.
    import cv2

    if (
        H is None
        or inlier_mask is None
        or int(inlier_mask.sum()) == 0
    ):
        return None

    # Transform the source inlier points into the reference coordinate
    # system using the fitted homography.
    proj = cv2.perspectiveTransform(
        mkpts_src[inlier_mask].reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    # Euclidean distance between the predicted and observed reference points.
    err = np.linalg.norm(
        proj - mkpts_ref[inlier_mask],
        axis=1
    )

    return float(np.median(err))


def quality_gate(
    n_inliers,
    inlier_ratio,
    min_inliers=config.MIN_INLIERS,
    min_inlier_ratio=config.MIN_INLIER_RATIO
):
    """
    Decide whether a fitted transformation has enough support to be trusted.

    Both conditions must be satisfied:
      1. There must be enough absolute inliers.
      2. A sufficient fraction of all matches must agree with the model.

    The gate is intentionally kept strict. If it fails, the correct response
    is normally to investigate the overlap region, scale conversion, or
    coordinate system rather than simply increasing the RANSAC threshold.
    """

    return bool(
        n_inliers >= min_inliers
        and inlier_ratio >= min_inlier_ratio
    )


def fit_and_evaluate(
    mkpts_src,
    mkpts_ref,
    use_piecewise=False,
    n_segments=config.PIECEWISE_N_SEGMENTS,
    ransac_thresh=config.RANSAC_REPROJ_THRESH_PX
):
    """
    Main entry point used by the pipeline for geometric fitting.

    Depending on `use_piecewise`, either a single global homography or a
    collection of along-track homographies is fitted.

    The result also includes the quality-gate decision so downstream code
    does not have to duplicate the acceptance logic.

    Returns:
        {
            "mode": "global" or "piecewise",
            "H": ndarray or None,
            "segments": list or None,
            "inlier_mask": ndarray,
            "n_inliers": int,
            "inlier_ratio": float,
            "registration_accepted": bool,
        }
    """

    if use_piecewise:
        # Fit independent transformations for the along-track bands.
        segments, inlier_mask = fit_piecewise(
            mkpts_src,
            mkpts_ref,
            n_segments,
            ransac_thresh
        )

        # The segment fits use disjoint groups of matches, so their inlier
        # counts can be summed to obtain the overall piecewise support.
        n_inliers = sum(
            s["n_inliers"] for s in segments
        )

        inlier_ratio = (
            n_inliers / len(mkpts_src)
            if len(mkpts_src)
            else 0.0
        )

        return {
            "mode": "piecewise",
            "H": None,
            "segments": segments,
            "inlier_mask": inlier_mask,
            "n_inliers": n_inliers,
            "inlier_ratio": inlier_ratio,
            "registration_accepted": quality_gate(
                n_inliers,
                inlier_ratio
            ),
        }

    # Default mode: assume the entire source/reference pair can be explained
    # by one projective transformation.
    H, inlier_mask, n_inliers, inlier_ratio = fit_homography(
        mkpts_src,
        mkpts_ref,
        ransac_thresh
    )

    return {
        "mode": "global",
        "H": H,
        "segments": None,
        "inlier_mask": inlier_mask,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "registration_accepted": quality_gate(
            n_inliers,
            inlier_ratio
        ),
    }


def reproj_errors(mkpts_src, mkpts_ref, H):
    """
    Calculate the Euclidean reprojection error for every match.

    Unlike the inlier-specific metrics above, this function evaluates all
    supplied correspondences. It is useful after a new transformation has
    been fitted when the caller wants to apply its own error threshold.
    """

    # Make sure the point arrays have the shape expected by OpenCV's
    # perspectiveTransform function.
    proj = cv2.perspectiveTransform(
        np.asarray(mkpts_src, dtype=np.float32).reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    return np.linalg.norm(
        proj - np.asarray(mkpts_ref),
        axis=1
    )


def uniform_select(
    mkpts_src,
    mkpts_ref,
    confidences,
    inlier_mask,
    ref_shape,
    grid=config.UNIFORM_GRID,
    max_per_cell=config.UNIFORM_MAX_PER_CELL
):
    """
    Enforce spatially uniform correspondence selection.

    First, only RANSAC inliers are considered. The reference image is then
    divided into grid cells, and the highest-confidence matches from each
    occupied cell are retained, up to `max_per_cell`.

    Unlike uniformity_coverage(), which only reports how well the points
    are distributed, this function actually changes the surviving set of
    correspondences.

    Returns:
        Integer indices into the original match arrays.
    """

    # Start with the geometrically trusted matches only.
    idx = np.where(
        np.asarray(inlier_mask, dtype=bool)
    )[0]

    if len(idx) == 0:
        return idx

    h, w = int(ref_shape[0]), int(ref_shape[1])
    gh, gw = int(grid[0]), int(grid[1])

    conf = np.asarray(confidences, dtype=float)

    # If the matcher did not provide a usable confidence value for every
    # point, treat all matches equally rather than failing the selection step.
    if len(conf) != len(mkpts_src):
        conf = np.ones(len(mkpts_src))

    # Convert each reference point into its corresponding grid cell.
    gx = np.clip(
        (
            np.asarray(mkpts_ref)[idx, 0]
            / max(w, 1)
            * gw
        ).astype(int),
        0,
        gw - 1
    )

    gy = np.clip(
        (
            np.asarray(mkpts_ref)[idx, 1]
            / max(h, 1)
            * gh
        ).astype(int),
        0,
        gh - 1
    )

    chosen = []

    # Visit every grid cell so that the final correspondence set cannot
    # become dominated by one highly textured part of the image.
    for cy in range(gh):
        for cx in range(gw):

            # Get the original match indices that fall inside this cell.
            ids = idx[
                (gx == cx)
                & (gy == cy)
            ]

            if len(ids):
                # Rank matches by confidence, highest first, and keep only
                # the requested number from this cell.
                order = ids[
                    np.argsort(conf[ids])[::-1]
                ]

                chosen.extend(
                    order[:max_per_cell].tolist()
                )

    return np.asarray(chosen, dtype=int)


def fit_uniform(
    mkpts_src,
    mkpts_ref,
    confidences,
    ref_shape,
    ransac_thresh=config.RANSAC_REPROJ_THRESH_PX
):
    """
    Fit -> spatially select -> refit -> recompute inliers.

    The first RANSAC fit provides a trustworthy starting set of inliers.
    Those inliers are then spatially filtered so that one small region does
    not dominate the final correspondence set.

    A second homography is fitted using the selected points, and the final
    RMSE, median error, and spatial coverage are calculated.

    Returns:
        H,
        final source points,
        final reference points,
        RMSE,
        median reprojection error,
        coverage.

    If the initial fit fails, H=None is returned and the caller can decide
    whether to fall back to another strategy.
    """

    # First establish which matches are geometrically consistent.
    H, mask, _, _ = fit_homography(
        mkpts_src,
        mkpts_ref,
        ransac_thresh
    )

    if H is None:
        return (
            None,
            np.empty((0, 2)),
            np.empty((0, 2)),
            None,
            None,
            0.0
        )

    # Keep the strongest inliers while enforcing spatial coverage.
    sel = uniform_select(
        mkpts_src,
        mkpts_ref,
        confidences,
        mask,
        ref_shape
    )

    # Refit after selection because removing spatially redundant points can
    # change the best-fitting transformation.
    H2, _, _, _ = fit_homography(
        mkpts_src[sel],
        mkpts_ref[sel],
        ransac_thresh
    )

    if H2 is not None:
        H = H2

        # Re-evaluate the selected points against the new transformation.
        # This prevents points that became inconsistent after refitting from
        # being included in the final result.
        err = reproj_errors(
            mkpts_src[sel],
            mkpts_ref[sel],
            H
        )

        keep = err <= ransac_thresh

        final_src = mkpts_src[sel][keep]
        final_ref = mkpts_ref[sel][keep]

    else:
        # If the second fit fails, retain the original RANSAC inliers instead
        # of throwing away an otherwise usable transformation.
        final_src = mkpts_src[mask]
        final_ref = mkpts_ref[mask]

    if len(final_src) == 0:
        return H, final_src, final_ref, None, None, 0.0

    # Calculate the final geometric error using the points that actually
    # survived the selection and refitting process.
    err = reproj_errors(
        final_src,
        final_ref,
        H
    )

    rmse = float(
        np.sqrt(np.mean(err ** 2))
    )

    median = float(
        np.median(err)
    )

    coverage = uniformity_coverage(
        final_ref,
        ref_shape
    )

    return (
        H,
        final_src,
        final_ref,
        rmse,
        median,
        coverage
    )


def refine_points(
    image_a,
    image_b,
    pts_a,
    pts_b,
    win=tuple(config.SUBPIXEL_WIN),
    max_shift=config.SUBPIXEL_MAX_SHIFT
):
    """
    Refine corresponding points to sub-pixel precision using cornerSubPix.

    Points too close to either image border are removed first because
    cornerSubPix needs a neighborhood around every point. Border keypoints
    are common when working with cropped tiles, so this filtering prevents
    OpenCV from failing on an otherwise valid match set.

    Returns the surviving refined point pairs. The number of points can be
    smaller than the input and may fall below the four-point minimum needed
    for another homography fit.
    """

    pts_a = np.asarray(pts_a, dtype=float)
    pts_b = np.asarray(pts_b, dtype=float)

    if len(pts_a) == 0:
        return pts_a, pts_b

    # Leave enough room around each point for the sub-pixel optimization
    # window. The +1 makes the boundary check conservative.
    m = max(1, int(win[0]) + 1)

    ha, wa = image_a.shape[:2]
    hb, wb = image_b.shape[:2]

    inside = (
        (pts_a[:, 0] >= m)
        & (pts_a[:, 0] <= wa - 1 - m)
        & (pts_a[:, 1] >= m)
        & (pts_a[:, 1] <= ha - 1 - m)
        & (pts_b[:, 0] >= m)
        & (pts_b[:, 0] <= wb - 1 - m)
        & (pts_b[:, 1] >= m)
        & (pts_b[:, 1] <= hb - 1 - m)
    )

    dropped = int(
        len(pts_a) - inside.sum()
    )

    if dropped:
        print(
            f"Sub-pixel refine: dropped "
            f"{dropped} border points (of {len(pts_a)})"
        )

    pts_a, pts_b = pts_a[inside], pts_b[inside]

    # There is no point attempting another homography fit if refinement has
    # left fewer than four point pairs.
    if len(pts_a) < 4:
        print(
            "Sub-pixel refine: too few interior points; "
            "keeping unrefined coordinates."
        )
        return pts_a, pts_b

    # Stop when either the requested accuracy is reached or the maximum
    # number of iterations has been exhausted.
    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-3
    )

    # cornerSubPix expects float32 arrays with shape (N, 1, 2).
    a = (
        pts_a.astype(np.float32)
        .reshape(-1, 1, 2)
        .copy()
    )

    b = (
        pts_b.astype(np.float32)
        .reshape(-1, 1, 2)
        .copy()
    )

    cv2.cornerSubPix(
        image_a,
        a,
        win,
        (-1, -1),
        criteria
    )

    cv2.cornerSubPix(
        image_b,
        b,
        win,
        (-1, -1),
        criteria
    )

    q0 = a.reshape(-1, 2)
    q1 = b.reshape(-1, 2)

    if max_shift is not None:
        # Reject refinements that moved a point by an unexpectedly large
        # amount. A large movement usually indicates that the original point
        # was not a stable corner for sub-pixel optimization.
        keep = (
            (np.linalg.norm(q0 - pts_a, axis=1) <= max_shift)
            & (np.linalg.norm(q1 - pts_b, axis=1) <= max_shift)
        )

        return q0[keep], q1[keep]

    return q0, q1


def refine_and_keep(
    image_a,
    image_b,
    pts_a,
    pts_b,
    H,
    ransac_thresh=config.RANSAC_REPROJ_THRESH_PX
):
    """
    Attempt sub-pixel refinement and keep it only when it improves RMSE.

    This function deliberately uses a conservative "never make things worse"
    policy. The original homography and points remain the fallback whenever
    refinement fails or produces a higher RMSE.

    Returns:
        H_out,
        refined/source points,
        refined/reference points,
        RMSE before refinement,
        RMSE after refinement (or None),
        whether refinement was accepted.
    """

    # Establish a baseline before changing the point coordinates.
    before = (
        float(
            np.sqrt(
                np.mean(
                    reproj_errors(pts_a, pts_b, H) ** 2
                )
            )
        )
        if len(pts_a)
        else float("inf")
    )

    # Sub-pixel refinement can be disabled through configuration. It also
    # makes no sense when there are too few points for another fit.
    if not config.SUBPIXEL_REFINE or len(pts_a) < 4:
        return (
            H,
            pts_a,
            pts_b,
            before,
            None,
            False
        )

    # Refine both sides of every correspondence.
    r0, r1 = refine_points(
        image_a,
        image_b,
        pts_a,
        pts_b
    )

    # Fit a fresh homography to the refined coordinates.
    H2, m2, n2, _ = fit_homography(
        r0,
        r1,
        ransac_thresh
    )

    if H2 is None or m2 is None or not m2.any():
        print(
            f"Sub-pixel refinement rejected: "
            f"RMSE {before:.4f} -> no fit"
        )

        return (
            H,
            pts_a,
            pts_b,
            before,
            None,
            False
        )

    # Evaluate only the new inliers from the refined fit.
    after = float(
        np.sqrt(
            np.mean(
                reproj_errors(
                    r0[m2],
                    r1[m2],
                    H2
                ) ** 2
            )
        )
    )

    # Accept the refinement only when it genuinely improves the error and
    # still leaves enough points to support the transformation.
    if after < before and n2 >= 4:
        print(
            f"Sub-pixel refinement accepted: "
            f"RMSE {before:.4f} -> {after:.4f} px"
        )

        return (
            H2,
            r0[m2],
            r1[m2],
            before,
            after,
            True
        )

    # Keeping the original result is safer than accepting a numerically
    # worse refinement just because the optimizer produced a new solution.
    print(
        f"Sub-pixel refinement rejected: "
        f"RMSE {before:.4f} -> {after:.4f} px"
    )

    return (
        H,
        pts_a,
        pts_b,
        before,
        after,
        False
    )


def coarse_ncc(
    image_a,
    image_b,
    valid_mask=None
):
    """
    Compute zero-mean normalized cross-correlation over valid pixels.

    NCC is used here only as a cheap diagnostic for coarse alignment. It can
    indicate whether two cropped regions have broadly similar structure,
    but it is not reliable enough to serve as the registration acceptance
    criterion, especially when illumination or sensor characteristics differ.
    """

    a = np.asarray(
        image_a,
        dtype=np.float32
    )

    b = np.asarray(
        image_b,
        dtype=np.float32
    )

    if valid_mask is None:
        valid_mask = np.ones(
            a.shape[:2],
            dtype=bool
        )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool
    )

    if not valid_mask.any():
        return 0.0

    # Remove the mean so the score measures structural similarity rather
    # than simply comparing absolute brightness levels.
    a = a - float(
        a[valid_mask].mean()
    )

    b = b - float(
        b[valid_mask].mean()
    )

    den = float(
        np.sqrt(
            float((a[valid_mask] ** 2).sum())
            * float((b[valid_mask] ** 2).sum())
        )
    )

    # A zero denominator means one of the valid regions has no variation.
    # In that case NCC is undefined, so return the neutral diagnostic value.
    if den <= 0:
        return 0.0

    return float(
        (a[valid_mask] * b[valid_mask]).sum()
        / den
    )


def metadata_corner_homography(
    source_meta,
    reference_meta,
    moon_radius_m=1737400.0
):
    """
    Build a coarse source-pixel -> reference-pixel homography directly from
    the geographic corner coordinates in the metadata.

    This is a fallback that does not require image pixels. The latitude and
    longitude corners are first converted into local metre coordinates using
    an equirectangular approximation around the mean latitude of the
    reference footprint.

    Four corner correspondences are enough to define a projective transform,
    so the method preserves the actual quadrilateral geometry rather than
    forcing the source and reference footprints to behave like similar
    rectangles.

    The resulting transform is intended for coarse ROI extraction. It is not
    a substitute for a rigorous sensor/camera model or full cartographic
    reprojection.

    Raises:
        ValueError if either image is missing its corner coordinates.
    """

    # Keep the same corner ordering for both datasets. Consistent ordering is
    # essential because the homography assumes each source corner corresponds
    # to the reference corner in the same position.
    order = [
        "upper_left",
        "upper_right",
        "lower_right",
        "lower_left"
    ]

    sc = source_meta.get("corners")
    rc = reference_meta.get("corners")

    if not sc or not rc:
        raise ValueError(
            "metadata_corner_homography needs corners on both sides"
        )

    # Build [latitude, longitude] pairs from the reference metadata.
    rl = np.array(
        [
            [
                rc[f"{k}_latitude"],
                rc[f"{k}_longitude"]
            ]
            for k in order
        ],
        dtype=float
    )

    # Use the mean reference latitude for the local equirectangular
    # conversion. This keeps the longitude scale approximately correct over
    # the relatively small footprint being considered.
    la0 = float(
        np.deg2rad(rl[:, 0].mean())
    )

    nl = int(reference_meta["shape"][0])
    ns = int(reference_meta["shape"][1])

    # Pixel coordinates of the four corners of the reference raster.
    pix = np.array(
        [
            [0, 0],
            [ns - 1, 0],
            [ns - 1, nl - 1],
            [0, nl - 1]
        ],
        dtype=np.float32
    )

    def to_meters(ll):
        """
        Convert local latitude/longitude coordinates to approximate lunar
        metre coordinates.

        Longitude is scaled by cos(latitude), while latitude is treated as
        a direct arc length. This is sufficient for the local coarse
        transformation used here.
        """

        return (
            np.stack(
                [
                    moon_radius_m
                    * np.cos(la0)
                    * np.deg2rad(ll[:, 1]),

                    moon_radius_m
                    * np.deg2rad(ll[:, 0])
                ],
                axis=1
            )
            .astype(np.float32)
        )

    # First construct a transform from geographic metre coordinates to
    # reference-image pixels.
    Hm = cv2.getPerspectiveTransform(
        to_meters(rl),
        pix
    )

    # Convert the source footprint corners into the same local metre frame
    # and project them into reference-image coordinates.
    sl = np.array(
        [
            [
                sc[f"{k}_latitude"],
                sc[f"{k}_longitude"]
            ]
            for k in order
        ],
        dtype=float
    )

    ref4 = cv2.perspectiveTransform(
        to_meters(sl).reshape(-1, 1, 2),
        Hm
    ).reshape(-1, 2).astype(np.float32)

    sh = int(source_meta["shape"][0])
    sw = int(source_meta["shape"][1])

    # Pixel coordinates of the source-image corners.
    src4 = np.array(
        [
            [0, 0],
            [sw - 1, 0],
            [sw - 1, sh - 1],
            [0, sh - 1]
        ],
        dtype=np.float32
    )

    # Finally map source pixels directly into reference pixels. This gives
    # the overlap stage a useful coarse prior before image-based matching
    # performs the final geometric alignment.
    return cv2.getPerspectiveTransform(
        src4,
        ref4
    ).astype(float)