"""
Stage 6 of the pipeline: feature matching.

This module handles the main correspondence stage:

    Source image + Reference image
              ↓
       SuperPoint + SuperGlue
              ↓
        candidate matches
              ↓
       geometric validation
              ↓
       reliable correspondences

SuperGlue works on image tiles rather than the entire lunar image at once.
That keeps memory usage manageable and also prevents a large image from
becoming too difficult for the feature matcher to handle.

SIFT is kept in sift_match.py as a fallback.

PyTorch and the official SuperGlue repository are imported lazily. This is
intentional: users who only want to run the SIFT pipeline should not need
PyTorch or the SuperGlue repository just to import this module.

Official SuperGlue repository:
https://github.com/magicleap/SuperGluePretrainedNetwork
"""

import time

import numpy as np

import LunarSync.config as config


def _torch_device():
    """
    Pick the fastest available PyTorch device.

    CUDA is preferred when a GPU is available; otherwise the matcher simply
    falls back to the CPU. Keeping this decision in one small helper avoids
    repeating the same device check throughout the module.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_matcher(
    weights=config.SUPERGLUE_WEIGHTS,
    max_keypoints=config.MAX_KEYPOINTS,
    keypoint_threshold=config.KEYPOINT_THRESHOLD,
    nms_radius=config.NMS_RADIUS,
    sinkhorn_iterations=config.SINKHORN_ITERATIONS,
    match_threshold=config.MATCH_THRESHOLD,
):
    """
    Build the SuperPoint + SuperGlue matcher.

    All of the important matching parameters come from config.py by default,
    so experiments can be tuned without editing this file.

    The actual SuperGlue classes are imported only when this function is
    called. That keeps SIFT-only or CPU-only parts of the pipeline lightweight.
    """

    # The official SuperGlue repository must be available on sys.path before
    # this function is called.
    from models.matching import Matching

    model_config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": weights,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }

    import torch

    # eval() disables training-specific behavior because this pipeline only
    # uses the pretrained network for inference.
    return (
        Matching(model_config)
        .eval()
        .to(_torch_device())
    )


def resize_long_side(img, max_side):
    """
    Shrink an image so its longest side is no larger than max_side.

    Returns:
        resized_image, scale

    The scale is returned because feature coordinates found on the resized
    image must later be converted back into the original image coordinates.
    """

    h, w = img.shape[:2]

    # This is the factor that converts original pixels into resized pixels.
    scale = max_side / max(h, w)

    # Don't enlarge images that are already small enough.
    if scale >= 1.0:
        return img, 1.0

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    import cv2

    # INTER_AREA is a good choice when reducing an image because it avoids
    # simply throwing away pixels independently.
    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def match_pair(matcher, gray0, gray1):
    """
    Match one pair of same-scale grayscale images with SuperGlue.

    The returned coordinates are expressed in the exact pixel coordinate
    system of the images passed into this function. If the caller resized
    the images first, the coordinates are therefore resized-image
    coordinates too.

    The caller is responsible for converting them back to full-resolution
    coordinates when necessary.
    """

    import torch
    from models.utils import frame2tensor

    device = _torch_device()

    # SuperGlue expects image tensors rather than raw NumPy arrays.
    tensor0 = frame2tensor(
        gray0.astype(np.float32),
        device,
    )

    tensor1 = frame2tensor(
        gray1.astype(np.float32),
        device,
    )

    # Normally the matcher is already in evaluation mode, but remember its
    # previous state so this helper does not unexpectedly change the caller's
    # model state.
    was_training = matcher.training
    matcher.eval()

    # No gradients are needed during registration. This saves memory and
    # avoids unnecessary computation, which matters when processing many
    # tiles.
    with torch.no_grad():
        prediction = matcher({
            "image0": tensor0,
            "image1": tensor1,
        })

    # Restore the previous state just in case the matcher was being used
    # elsewhere in a different mode.
    if was_training:
        matcher.train()

    # Move the model outputs from GPU/torch tensors back into NumPy arrays.
    prediction = {
        k: v[0].cpu().numpy()
        for k, v in prediction.items()
    }

    kpts0 = prediction["keypoints0"]
    kpts1 = prediction["keypoints1"]

    matches = prediction["matches0"]
    confidence = prediction["matching_scores0"]

    # SuperGlue uses -1 to indicate that a detected keypoint has no valid
    # correspondence in the other image.
    valid = matches > -1

    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = confidence[valid]

    return mkpts0, mkpts1, mconf


def match_tiled(
    matcher,
    source_img,
    reference_img,
    tile_px=config.TILE_PX,
    overlap_px=config.TILE_OVERLAP_PX,
    resize_long_side_px=config.RESIZE_LONG_SIDE,
    skip_uniform_std=config.SKIP_UNIFORM_TILE_STD,
    verbose=True,
):
    """
    Match the source and reference images tile by tile.

    Instead of giving SuperGlue a potentially huge lunar image, we divide the
    common image area into overlapping tiles and process each tile separately.

    This helps in two ways:

    1. Memory stays manageable.
    2. Local terrain gets its own feature-matching context.

    The function assumes that the source and reference have already been
    brought to a common scale and shape. The actual geometric alignment is
    still performed later by the registration stages.

    Returns:
        mkpts_src
        mkpts_ref
        confidences

    All returned coordinates are converted back into the original
    source/reference image coordinate systems.
    """

    h, w = reference_img.shape[:2]

    # Overlapping tiles give features near a tile boundary another chance to
    # appear in a neighboring tile. The 256-pixel lower bound prevents an
    # accidentally tiny step from creating an enormous number of tiles.
    step = max(
        256,
        tile_px - overlap_px,
    )

    all_mkpts_src = []
    all_mkpts_ref = []
    all_conf = []

    # This is only for progress reporting; it does not control the matching.
    n_tiles_total = (
        ((h + step - 1) // step)
        * ((w + step - 1) // step)
    )

    tile_i = 0
    n_skipped = 0
    start_time = time.time()

    for y0 in range(0, h, step):
        y1 = min(y0 + tile_px, h)

        for x0 in range(0, w, step):
            x1 = min(x0 + tile_px, w)

            tile_i += 1

            # Crop the same nominal region from both images. This relies on
            # the caller having already performed the coarse alignment and
            # shape normalization.
            ref_tile = reference_img[y0:y1, x0:x1]
            src_tile = source_img[y0:y1, x0:x1]

            # Edge tiles can occasionally end up with different dimensions.
            # SuperGlue expects a proper pair, so simply skip those cases.
            if src_tile.shape != ref_tile.shape or src_tile.size == 0:
                continue

            # A nearly uniform tile usually contains little useful structure:
            # for example, a flat shadow, blank region, or very low-texture
            # terrain. There is little value in spending GPU time on it.
            if float(np.std(ref_tile)) < skip_uniform_std:
                n_skipped += 1
                continue

            # SuperGlue doesn't need the full resolution of every tile.
            # Resizing makes the matcher substantially cheaper while still
            # retaining enough structure for correspondence detection.
            src_resized, src_scale = resize_long_side(
                src_tile,
                resize_long_side_px,
            )

            ref_resized, ref_scale = resize_long_side(
                ref_tile,
                resize_long_side_px,
            )

            mkpts0, mkpts1, mconf = match_pair(
                matcher,
                src_resized,
                ref_resized,
            )

            # SuperGlue returned coordinates relative to the resized tile.
            # First undo the resize, then add the tile's (x0, y0) offset to
            # put everything back into full-image coordinates.
            mkpts0_full = (
                mkpts0 / src_scale
                + np.array([x0, y0])
            )

            mkpts1_full = (
                mkpts1 / ref_scale
                + np.array([x0, y0])
            )

            all_mkpts_src.append(mkpts0_full)
            all_mkpts_ref.append(mkpts1_full)
            all_conf.append(mconf)

            # Print occasional progress instead of producing one line for
            # every tile. This keeps long Colab runs readable.
            if verbose and (
                tile_i % 25 == 0
                or tile_i == n_tiles_total
            ):
                elapsed = time.time() - start_time

                n_matches_so_far = sum(
                    len(m)
                    for m in all_mkpts_src
                )

                print(
                    f"  tile {tile_i}/{n_tiles_total}  "
                    f"({elapsed:.0f}s elapsed, "
                    f"{n_matches_so_far} matches so far, "
                    f"{n_skipped} tiles skipped as uniform)"
                )

    # Keep the return type predictable even when no tile produced matches.
    if not all_mkpts_src:
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty((0,)),
        )

    return (
        np.concatenate(all_mkpts_src),
        np.concatenate(all_mkpts_ref),
        np.concatenate(all_conf),
    )


def match_with_fallback(
    matcher,
    enhanced_source,
    enhanced_reference,
    verbose=True,
):
    """
    Run SuperGlue first and fall back to SIFT when necessary.

    SuperGlue is the primary matcher, while SIFT provides a useful classical
    backup for difficult scenes where the learned matcher does not return
    enough raw correspondences.

    Importantly, this function only produces candidate correspondences.
    It does NOT decide whether the estimated geometry is trustworthy.
    RANSAC and the downstream quality checks make that decision later.
    """

    # Primary path: learned feature detection + matching.
    mkpts_src, mkpts_ref, conf = match_tiled(
        matcher,
        enhanced_source,
        enhanced_reference,
        verbose=verbose,
    )

    # If SuperGlue already produced enough candidates, there is no reason to
    # spend additional time running the fallback.
    if len(mkpts_src) >= config.FALLBACK_MIN_RAW_MATCHES:
        return (
            mkpts_src,
            mkpts_ref,
            conf,
            "superglue",
        )

    # Import SIFT only when we actually need it. This keeps SuperGlue-only
    # environments from unnecessarily loading the SIFT implementation.
    from LunarSync.sift_match import match_tiled_sift

    s_src, s_ref, s_conf = match_tiled_sift(
        enhanced_source,
        enhanced_reference,
        verbose=verbose,
    )

    # At this stage we compare raw candidate counts only. A larger set is not
    # automatically a better set -- geometric validation happens downstream.
    if len(s_src) > len(mkpts_src):
        if verbose:
            print(
                f"Fallback: SIFT wins on raw count "
                f"({len(s_src)} vs {len(mkpts_src)})."
            )

        return (
            s_src,
            s_ref,
            s_conf,
            "sift",
        )

    if verbose:
        print(
            f"Fallback: SIFT tried ({len(s_src)} raw) but "
            f"SuperGlue kept ({len(mkpts_src)} raw)."
        )

    return (
        mkpts_src,
        mkpts_ref,
        conf,
        "superglue_and_sift_tied",
    )


def match_fine(
    matcher,
    coarse_warped,
    reference,
    use_fallback=True,
    verbose=True,
):
    """
    Perform the fine correspondence stage after coarse alignment.

    The normal order is:

        SuperGlue
             ↓
        SIFT fallback
             ↓
        relaxed-SIFT retry if still starved

    The relaxed SIFT pass is useful for difficult lunar imagery where the
    normal Lowe ratio is too strict. It is only attempted when the first
    matching pass produces too few candidates.

    As with match_with_fallback(), this function does not accept or reject
    the final transformation. That decision belongs to geometry_fit.py.
    """

    if matcher is None or not use_fallback:

        # matcher=None means the caller explicitly wants a SIFT-only run.
        if matcher is None:
            from LunarSync.sift_match import match_tiled_sift

            mkpts_src, mkpts_ref, conf = match_tiled_sift(
                coarse_warped,
                reference,
                verbose=verbose,
            )

            matcher_used = "sift"

        else:
            # SuperGlue is available and the caller has disabled fallback.
            mkpts_src, mkpts_ref, conf = match_tiled(
                matcher,
                coarse_warped,
                reference,
                verbose=verbose,
            )

            matcher_used = "superglue"

    else:
        # Normal production path: SuperGlue first, SIFT if the result is weak.
        mkpts_src, mkpts_ref, conf, matcher_used = match_with_fallback(
            matcher,
            coarse_warped,
            reference,
            verbose=verbose,
        )

    # Don't bother with another expensive attempt if we already have enough
    # candidates for the geometric fitting stage.
    need = max(
        config.MIN_INLIERS,
        25,
    )

    if (
        len(mkpts_src) < need
        and matcher_used != "sift-relaxed"
    ):
        from LunarSync.sift_match import match_tiled_sift

        # A relaxed Lowe ratio allows more potentially useful correspondences
        # through. The trade-off is that it can also admit more false matches,
        # which is why geometric validation must happen afterwards.
        s_src, s_ref, s_conf = match_tiled_sift(
            coarse_warped,
            reference,
            lowe_ratio=config.FINE_SIFT_RELAX_RATIO,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Relaxed-SIFT retry "
                f"(ratio={config.FINE_SIFT_RELAX_RATIO}): "
                f"{len(s_src)} raw."
            )

        # Keep the larger candidate set. Again, this is only a raw-count
        # decision; RANSAC will determine which matches are actually useful.
        if len(s_src) > len(mkpts_src):
            if verbose:
                print(
                    f"Kept relaxed set "
                    f"({len(s_src)} vs {len(mkpts_src)})."
                )

            mkpts_src = s_src
            mkpts_ref = s_ref
            conf = s_conf
            matcher_used = "sift-relaxed"

    return (
        mkpts_src,
        mkpts_ref,
        conf,
        matcher_used,
    )


def match_coarse_sift(
    source_preview,
    reference_roi_preview,
    gsd_scale,
    levels=tuple(config.COARSE_PYRAMID_LEVELS),
    min_short_side=config.COARSE_MIN_SHORT_SIDE,
    upscale_cap=config.COARSE_UPSCALE_CAP,
    max_features=8000,
    verbose=True,
):
    """
    Use SIFT to get a coarse estimate of where the source belongs in the
    reference ROI.

    The important part here is the scale pyramid. OHRC, TMC-2 and NAC can have
    very different pixel resolutions, and the previews may also have been
    generated with different decimation factors.

    Instead of trusting one scale factor, several nearby scale hypotheses are
    tested. Each hypothesis gets a geometric score, and the strongest
    consensus wins.

    This stage only needs enough reliable matches to establish a coarse
    location. If SIFT cannot find sufficient consensus, the caller can fall
    back to the metadata-derived corner homography.
    """

    import cv2
    import LunarSync.geometry_fit as geometry_fit
    from LunarSync.sift_match import match_tile_pair, build_sift

    sift = build_sift()

    best = None

    for j in levels:

        # Move through octave-like scale levels around the expected GSD ratio.
        # This compensates for the fact that preview images may not have the
        # exact same effective resolution.
        k = gsd_scale * (2.0 ** j)

        nw = max(
            1,
            int(round(source_preview.shape[1] * k)),
        )

        nh = max(
            1,
            int(round(source_preview.shape[0] * k)),
        )

        u = 1.0

        # If the candidate source image becomes too small, temporarily
        # upscale both sides enough to keep feature detection meaningful.
        need = min(
            nw,
            nh,
            reference_roi_preview.shape[0],
            reference_roi_preview.shape[1],
        )

        if need < min_short_side:
            u = min(
                upscale_cap,
                min_short_side / need,
            )

        # Effective scale after any safety upscaling.
        ke = k * u

        src_try = cv2.resize(
            source_preview,
            (
                max(
                    1,
                    int(round(source_preview.shape[1] * ke)),
                ),
                max(
                    1,
                    int(round(source_preview.shape[0] * ke)),
                ),
            ),
            interpolation=(
                cv2.INTER_AREA
                if ke < 1.0
                else cv2.INTER_LINEAR
            ),
        )

        # When we upscaled the source, also upscale the reference ROI so that
        # the two images remain comparable at this pyramid level.
        if u > 1.0:
            ref_try = cv2.resize(
                reference_roi_preview,
                None,
                fx=u,
                fy=u,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            ref_try = reference_roi_preview

        # Run the classical SIFT matcher at this particular scale hypothesis.
        m0, m1, _ = match_tile_pair(
            sift,
            src_try,
            ref_try,
        )

        # A set of raw matches is not enough. Fit a homography and use its
        # inlier count as the main indication that the matches agree on a
        # plausible geometric relationship.
        Hc, mc, ni, ir = geometry_fit.fit_homography(
            m0,
            m1,
        )

        if verbose:
            if mc is not None:
                print(
                    f"  coarse level j={j:+d} "
                    f"k={ke:.4f} "
                    f"shapes={src_try.shape} vs {ref_try.shape}: "
                    f"raw={len(m0)} "
                    f"inliers={ni} "
                    f"ratio={ir:.3f}"
                )
            else:
                print(
                    f"  coarse level j={j:+d} "
                    f"k={ke:.4f}: "
                    f"raw={len(m0)} no fit"
                )

        # Keep the scale that gives the strongest geometric consensus.
        if (
            Hc is not None
            and (
                best is None
                or ni > best["n_inliers"]
            )
        ):
            best = {
                "n_inliers": ni,
                "k_eff": ke,
                "u": u,
                "src_try": src_try,
                "ref_try": ref_try,
                "mkpts_src": m0,
                "mkpts_ref": m1,
                "Hc": Hc,
                "mask": mc,
                "inlier_ratio": ir,
            }

    # Coarse localization is only considered successful when there is enough
    # geometric consensus. A weak fit is treated as "no reliable coarse fit"
    # so the caller can use the metadata-based fallback instead.
    if (
        best is None
        or best["n_inliers"] < config.COARSE_MIN_INLIERS
    ):
        return (
            None,
            source_preview,
            reference_roi_preview,
        )

    return (
        best,
        best["src_try"],
        best["ref_try"],
    )