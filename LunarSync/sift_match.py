"""
Stage 6B of the registration pipeline.

SIFT is used here as the fallback matcher when the primary SuperGlue
matcher (Stage 6A) does not meet the quality threshold.

Why keep SIFT as a fallback?
    SIFT is relatively simple, well understood, and works well on
    less-distorted lunar image pairs, particularly away from difficult
    polar/low-contrast regions. It is not expected to be the best choice
    for every lunar scene, which is why SuperGlue remains the primary
    matcher.

The matcher expects the images to have already gone through the common
preprocessing pipeline. In particular, the source and reference images
should already be brought to a comparable scale and enhanced before
feature extraction.
"""

import numpy as np

import LunarSync.config as config


def build_sift():
    """
    Construct the SIFT detector using the parameters defined in config.py.
    Keeping the parameters in one configuration file makes the matcher
    easier to reproduce and avoids silently changing feature-extraction
    behaviour in different parts of the pipeline.
    """
    import cv2
    try:
        sift = cv2.SIFT_create(
            nfeatures=config.SIFT_N_FEATURES,
            nOctaveLayers=config.SIFT_N_OCTAVE_LAYERS,
            contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=config.SIFT_EDGE_THRESHOLD,
        )
    except AttributeError as exc:
        # SIFT is included in modern OpenCV builds, but older installations
        # may not expose it. Fail with a useful message instead of producing
        # a confusing error later during feature extraction.
        raise ImportError(
            "This OpenCV build has no SIFT (need opencv-contrib or "
            "opencv-python>=4.4 with SIFT reinstated)."
        ) from exc
    return sift


def match_descriptors_knn(desc0, desc1, lowe_ratio=config.SIFT_LOWE_RATIO):
    """
    Match two descriptor sets using kNN matching followed by Lowe's
    ratio test.

    Each descriptor in desc0 is compared with its two nearest neighbours
    in desc1. A match is accepted only when the best neighbour is
    sufficiently better than the second-best neighbour.

    This helps reject ambiguous descriptors, which is particularly useful
    on repetitive lunar terrain where similar-looking local structures
    can occur close together.

    Returns:
        List of (descriptor_index_in_desc0, descriptor_index_in_desc1)
        pairs that pass the ratio test.
    """
    import cv2

    # There is no meaningful two-neighbour comparison when either descriptor
    # set is too small. Returning an empty list lets the caller handle this
    # tile exactly like a tile with no valid features.
    if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc0, desc1, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair

        # Lowe's ratio test measures how distinctive the best match is
        # compared with the next-best candidate.
        if m.distance < lowe_ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good


def apply_cross_check(pairs_fwd, pairs_rev):
    """
    Keep only matches that are consistent in both directions.

    A forward match says:
        descriptor A -> descriptor B

    The reverse match must independently say:
        descriptor B -> descriptor A

    Requiring this mutual agreement removes a useful class of accidental
    one-way matches before the more expensive geometric filtering stage.

    """

    # Reverse pairs are stored as (reference_idx, source_idx), so swap
    # them back into source/reference order before checking membership.
    rev_set = set((b, a) for a, b in pairs_rev)
    return [(a, b) for a, b in pairs_fwd if (a, b) in rev_set]


def match_tile_pair(sift, tile_src, tile_ref, lowe_ratio=config.SIFT_LOWE_RATIO):
    """
    Match one source/reference tile pair.

    Feature coordinates returned here are tile-local. The caller converts
    them back to full-image coordinates after this function returns.

    Returns:
        mkpts_src_tile : source keypoint coordinates
        mkpts_ref_tile : reference keypoint coordinates
        conf_tile      : descriptor-based confidence values

    The confidence is derived from the Lowe ratio:
        confidence = 1 - (best_distance / second_best_distance)

    A smaller Lowe ratio therefore corresponds to a more distinctive match.

    Note:
        Geometry fitting is deliberately kept outside this function.
        RANSAC/MAGSAC and the final inlier decision belong to
        geometry_fit.py, where all matches can be evaluated together.
    """
    import cv2

    # SIFT extracts both the keypoint locations and their descriptors.
    # If either side contains no usable descriptors, this tile contributes
    # nothing to the global match set.
    kp0, desc0 = sift.detectAndCompute(tile_src, None)
    kp1, desc1 = sift.detectAndCompute(tile_ref, None)

    if desc0 is None or desc1 is None:
        return (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))

    # First find forward matches: source -> reference.
    fwd = match_descriptors_knn(desc0, desc1, lowe_ratio)

    # An optional reverse pass makes the matcher stricter by requiring
    # the correspondence to survive in both directions.
    if config.SIFT_CROSS_CHECK:
        rev = match_descriptors_knn(desc1, desc0, lowe_ratio)
        pairs = apply_cross_check(fwd, rev)
    else:
        pairs = fwd

    if not pairs:
        return (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))

    # Re-run the small descriptor search so that we can recover the
    # second-neighbour distance needed to report a confidence score.
    #
    # This does not change which matches are accepted; it only gives the
    # reporting stage a measure of how distinctive each accepted match was.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn = bf.knnMatch(desc0, desc1, k=2)

    ratio_of = {}

    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            ratio_of[m.queryIdx] = m.distance / (n.distance + 1e-12)

    pts0, pts1, conf = [], [], []

    for a, b in pairs:
        # SIFT keypoint coordinates are floating-point coordinates already,
        # so we preserve them rather than rounding to integer pixels.
        # This retains the positional precision provided by SIFT's
        # keypoint localization.
        pts0.append(kp0[a].pt)   
        pts1.append(kp1[b].pt)

        # Convert the Lowe ratio into a more intuitive "higher is better"
        # confidence value for reporting.
        conf.append(1.0 - float(ratio_of.get(a, 1.0)))

    return np.array(pts0), np.array(pts1), np.array(conf)


def match_tiled_sift(
    source_img,
    reference_img,
    tile_px=config.TILE_PX,
    overlap_px=config.TILE_OVERLAP_PX,
    skip_uniform_std=config.SKIP_UNIFORM_TILE_STD,
    lowe_ratio=config.SIFT_LOWE_RATIO,
    verbose=True,
):
    """
    Run SIFT over the image pair using overlapping tiles.

    Why tiled matching?
        OHRC images can be extremely large, so running SIFT over the entire
        frame at once is unnecessarily expensive in both memory and compute.

        More importantly, processing local regions prevents the matcher from
        concentrating all correspondences in one visually distinctive area.
        The resulting matches can later be filtered/selected for better
        spatial coverage.

    Assumptions:
        - source and reference have already been brought to a common scale
        - both images have compatible dimensions
        - inputs are 8-bit images
        - preprocessing/enhancement has already been applied

    These conditions correspond to the preceding resampling and enhancement
    stages in the registration pipeline.

    Returns:
        mkpts_src    : source points in full-image coordinates
        mkpts_ref    : reference points in full-image coordinates
        confidences  : descriptor-based confidence for each match
    """
    sift = build_sift()

    h, w = reference_img.shape[:2]

    # The overlap means a feature close to a tile boundary has a chance
    # to appear in a neighbouring tile as well. This reduces the chance
    # that useful features are lost purely because of tile boundaries.
    step = max(256, tile_px - overlap_px)

    all_s, all_r, all_c = [], [], []
    n_skipped = 0
    for y0 in range(0, h, step):
        y1 = min(y0 + tile_px, h)

        for x0 in range(0, w, step):
            x1 = min(x0 + tile_px, w)

            ref_tile = reference_img[y0:y1, x0:x1]
            src_tile = source_img[y0:y1, x0:x1]

            # This should normally be guaranteed by the preprocessing
            # stages, but checking here makes the matcher safer when used
            # independently.
            if src_tile.shape != ref_tile.shape or src_tile.size == 0:
                continue

            # Nearly uniform tiles contain little useful texture and can
            # generate unstable or meaningless keypoints. This is common
            # in large shadowed or otherwise low-information lunar regions.
            if float(np.std(ref_tile)) < skip_uniform_std:
                n_skipped += 1
                continue

            m0, m1, mc = match_tile_pair(sift, src_tile, ref_tile, lowe_ratio)
            if len(m0) == 0:
                continue

            # match_tile_pair returns coordinates relative to the current
            # tile. Add the tile origin so every point is expressed in the
            # common full-image coordinate system expected by the geometry
            # fitting stage.
            m0 = m0 + np.array([x0, y0])
            m1 = m1 + np.array([x0, y0])
            all_s.append(m0)
            all_r.append(m1)
            all_c.append(mc)
    if verbose:
        n = sum(len(m) for m in all_s)
        print(f"SIFT tiled: {n} raw matches ({n_skipped} uniform tiles skipped).")

    # No valid correspondence survived any tile.
    if not all_s:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
    
    # Merge all tile-local results into one global correspondence set.    
    return np.concatenate(all_s), np.concatenate(all_r), np.concatenate(all_c)
