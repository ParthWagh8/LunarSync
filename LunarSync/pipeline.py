"""
Main orchestrator for the lunar image-registration pipeline.

This file intentionally contains very little image-processing logic.
Its job is to:

    1. Load metadata and image pixels
    2. Verify that the source/reference pair can be compared
    3. Estimate a coarse source-to-reference alignment
    4. Extract the likely overlap region
    5. Normalize the two images to a common physical scale
    6. Enhance them for feature matching
    7. Find and filter fine correspondences
    8. Estimate the final geometric transformation
    9. Warp the source into the reference frame
   10. Evaluate the final registration and build a report

Keeping the stages separated makes it easier to debug individual parts
of the registration process and replace a matcher or geometric model
without rewriting the whole pipeline.

Example:

    import pipeline

    result = pipeline.register_pair(
        source_dir="/content/drive/MyDrive/ohrc/.../data/calibrated/20210401",
        sensor="ohrc",
        nac_dir="/content/drive/MyDrive/nac",
        matcher=matching.load_matcher(),
    )

`result` contains both intermediate products useful for visualization
and the final `report` used for evaluation and saving.
"""

import math
from pathlib import Path

import cv2
import numpy as np

import LunarSync.config as config
import LunarSync.io_utils as io_utils
import LunarSync.metadata as metadata
import LunarSync.georeference as georeference
import LunarSync.overlap as overlap
import LunarSync.resample as resample
import LunarSync.enhance as enhance
import LunarSync.matching as matching
import LunarSync.geometry_fit as geometry_fit
import LunarSync.warp as warp
import LunarSync.evaluate as evaluate


def _read_roi(array, full_stride, bbox_full):
    """
    Extract a region defined in full-resolution coordinates.

    The array passed here may be a reduced-resolution preview rather than
    the original image. `full_stride` tells us how many original pixels
    are represented by one pixel in the current array.

    This keeps ROI coordinates consistent throughout the pipeline:
        full-resolution coordinates -> preview/native array coordinates

    Returns:
        roi              : extracted image region
        origin_raw       : full-resolution coordinate of roi[0, 0]
        clipped_bbox_full: actual full-resolution bounds that were read
    """

    sx, sy = float(full_stride[0]), float(full_stride[1])
    h, w = array.shape[:2]

    # Convert the requested full-resolution bounding box into coordinates
    # of the array we actually have in memory.
    qx0 = max(
        0,
        int(bbox_full[0] / sx)
    )
    qy0 = max(
        0,
        int(bbox_full[1] / sy)
    )

    qx1 = min(
        w,
        max(qx0 + 1, int(math.ceil(bbox_full[2] / sx)))
    )
    qy1 = min(
        h,
        max(qy0 + 1, int(math.ceil(bbox_full[3] / sy)))
    )

    # Convert the array-space origin back into full-resolution coordinates
    # so later homography calculations know exactly where this ROI came from.
    origin_x = qx0 * sx
    origin_y = qy0 * sy

    return (
        np.asarray(array[qy0:qy1, qx0:qx1]),
        (origin_x, origin_y),
        (origin_x, origin_y, qx1 * sx, qy1 * sy)
    )


def _project_corners(H, w, h):
    """
    Project the four corners of an image through a homography.

    This is mainly used to estimate where the source image lands inside
    the reference frame before extracting the fine-registration ROI.
    """

    corners = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ],
        dtype=np.float32
    )

    return cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)


def _clamp_bbox(x0, y0, x1, y1, w, h, pad=0):
    """
    Expand a bounding box by `pad` pixels and keep it inside the image.

    A small padding region is useful because the coarse transformation
    is only an approximate localization. Giving the fine matcher some
    extra context makes it less sensitive to small coarse-stage errors.
    """

    return (
        max(0, int(math.floor(x0 - pad))),
        max(0, int(math.floor(y0 - pad))),
        min(w, int(math.ceil(x1 + pad))),
        min(h, int(math.ceil(y1 + pad)))
    )


def _discover_preview(
    search_root,
    sensor,
    is_reference,
    reference_meta=None
):
    """
    Best-effort search for a preview PNG.

    Preview files are useful for lightweight/staged demonstrations, but
    their absence should not prevent a normal full-resolution run.
    Therefore this helper deliberately returns None rather than raising.
    """

    root = Path(search_root)

    if is_reference:
        cands = sorted(
            root.rglob("*.png")
        )

        key = ""

        if reference_meta is not None:
            pid = reference_meta.get("img_file") or ""
            key = pid.lower().split("_meta")[0]

        # Prefer a preview whose filename matches the reference product ID.
        for p in cands:
            if key and key in p.name.lower():
                return p

        # If an exact product match is unavailable, look for a likely
        # LROC preview before falling back to the first PNG we find.
        for p in cands:
            if (
                "nac" in str(p).lower()
                or "wac" in str(p).lower()
            ):
                return p

        return cands[0] if cands else None

    # Source previews use a short sensor-specific filename hint.
    key = {
        "ohrc": "ohr",
        "tmc": "tmc",
        "iirs": "iir"
    }.get(sensor, "")

    cands = [
        p
        for p in sorted(root.rglob("*.png"))
        if key in p.name.lower()
        and "_brw_" in p.name.lower()
    ]

    # Some datasets may not follow the expected browse naming convention,
    # so use a broader filename search as a fallback.
    if not cands:
        cands = [
            p
            for p in sorted(root.rglob("*.png"))
            if key in p.name.lower()
        ]

    return cands[0] if cands else None


def register_pair(
    source_dir,
    sensor,
    nac_dir,
    matcher,
    use_piecewise=False,
    verbose=True,
    reference_sensor="nac",
    use_fallback=True,
    source_preview_png=None,
    reference_preview_png=None,
    use_previews_only=False,
    enhance_recipe="paper"
):
    """
    Run one source/reference pair through the complete registration
    pipeline.

    Supported source sensors:
        "ohrc", "tmc", "iirs"

    Supported reference sensors:
        "nac", "wac"

    Matching strategy:
        SuperGlue is used as the primary fine matcher when available.
        SIFT can be enabled as a fallback for cases where SuperGlue
        does not pass its quality gate.

    The pipeline follows a coarse-to-fine strategy:

        metadata
          ↓
        pixels
          ↓
        coarse localization
          ↓
        overlap ROI
          ↓
        common GSD
          ↓
        enhancement
          ↓
        fine matching
          ↓
        geometric fitting
          ↓
        uniform point selection
          ↓
        refinement
          ↓
        final warp
          ↓
        evaluation

    `use_previews_only=True` is intended for staged demonstrations and
    debugging. The normal path uses the original `.img` products.

    Returns:
        A dictionary containing intermediate arrays, transformations,
        match points, fitted geometry, warped output, and the final report.
    """

    timer = evaluate.StageTimer()

    # Give every registration run a readable identifier. This ends up in
    # the final report and makes it easier to compare multiple experiments.
    pair_name = (
        f"{sensor}_"
        f"{source_dir.split('/')[-1] if isinstance(source_dir, str) else source_dir.name}"
    )

    # -------------------------------------------------------------------------
    # Stage 0: metadata
    # -------------------------------------------------------------------------
    timer.start("stage0_metadata")

    if use_previews_only:
        # In preview mode we still use the original metadata because the
        # preview itself does not contain the sensor geometry information
        # needed by later stages.
        source_meta_path = next(
            Path(source_dir).rglob("*.xml"),
            None
        )

        if source_meta_path is None:
            raise FileNotFoundError(
                f"No PDS4 label under {source_dir}"
            )

        nac_meta_path = next(
            Path(nac_dir).rglob("*_meta.json"),
            None
        )

        if nac_meta_path is None:
            raise FileNotFoundError(
                f"No meta.json under {nac_dir}"
            )

        source_meta = metadata.load_metadata(
            sensor,
            source_meta_path
        )

        reference_meta = metadata.load_metadata(
            reference_sensor,
            nac_meta_path
        )

        # Explicit preview paths take priority. Otherwise try to locate one
        # automatically from the dataset directory.
        if source_preview_png is None:
            source_preview_png = _discover_preview(
                source_dir,
                sensor,
                False
            )

        if reference_preview_png is None:
            reference_preview_png = _discover_preview(
                nac_dir,
                sensor,
                True,
                reference_meta
            )

        if (
            source_preview_png is None
            or reference_preview_png is None
        ):
            raise FileNotFoundError(
                "Preview-only run needs a source and a reference PNG."
            )

        if verbose:
            print(
                "Pixel mode: PREVIEW-PNG staged demo "
                f"({Path(source_preview_png).name} vs "
                f"{Path(reference_preview_png).name})"
            )

    else:
        # Normal production path: locate the actual calibrated image
        # products and their metadata labels.
        source_img_path, source_meta_path = (
            io_utils.find_product_files(
                source_dir,
                sensor
            )
        )

        nac_img_path, nac_meta_path = (
            io_utils.find_product_files(
                nac_dir,
                reference_sensor
            )
        )

        source_meta = metadata.load_metadata(
            sensor,
            source_meta_path
        )

        reference_meta = metadata.load_metadata(
            reference_sensor,
            nac_meta_path
        )

        if verbose:
            print("Pixel mode: full-.img")

    if verbose:
        metadata.print_geometry_summary(
            source_meta,
            sensor.upper()
        )

        metadata.print_geometry_summary(
            reference_meta,
            reference_sensor.upper()
        )

    timer.stop("stage0_metadata")

    # -------------------------------------------------------------------------
    # Stage 1: common frame / geometry check
    # -------------------------------------------------------------------------
    timer.start("stage1_georeference")

    # Before doing any expensive image processing, make sure the two
    # products have enough geographic information to establish a common
    # lunar frame.
    geometry_check = georeference.verify_common_frame(
        source_meta,
        reference_meta
    )

    if not geometry_check["ok"]:
        raise ValueError(
            f"Stage 1 failed: {geometry_check['note']}"
        )

    timer.stop("stage1_georeference")

    # -------------------------------------------------------------------------
    # Stage 2: load image pixels
    # -------------------------------------------------------------------------
    timer.start("stage2_pixels")

    src_band = None

    if use_previews_only:
        # Preview loaders also return the relationship between preview
        # coordinates and the original full-resolution image.
        src_arr, ssx, ssy = io_utils.load_preview_png(
            source_preview_png,
            source_meta["shape"]
        )

        ref_arr, rsx, rsy = io_utils.load_preview_png(
            reference_preview_png,
            reference_meta["shape"]
        )

        src_stride = (ssx, ssy)
        ref_stride = (rsx, rsy)

        src_pixels = (
            f"preview:{Path(source_preview_png).name}"
        )
        ref_pixels = (
            f"preview:{Path(reference_preview_png).name}"
        )

    else:
        # Full products may be very large, so io_utils handles the actual
        # product layout/offset details instead of duplicating them here.
        src_arr, src_band = io_utils.load_source_pixels(
            source_img_path,
            source_meta["shape"],
            source_meta["dtype"],
            source_meta["offset"]
        )

        ref_arr = io_utils.load_pixel_array(
            nac_img_path,
            reference_meta["shape"],
            reference_meta["dtype"],
            reference_meta["offset"]
        )

        # Full resolution arrays have a one-to-one relationship with the
        # original product coordinates.
        src_stride = (1.0, 1.0)
        ref_stride = (1.0, 1.0)

        src_pixels = (
            f"img:{Path(source_img_path).name}"
        )
        ref_pixels = (
            f"img:{Path(nac_img_path).name}"
        )

    if verbose:
        print(
            f"Source pixels: {src_pixels} "
            f"{src_arr.shape} strides={src_stride}"
            + (
                f" band={src_band}"
                if src_band is not None
                else ""
            )
        )

        print(
            f"Reference pixels: {ref_pixels} "
            f"{ref_arr.shape} strides={ref_stride}"
        )

    timer.stop("stage2_pixels")

    # -------------------------------------------------------------------------
    # Stage 2B: coarse localization
    # -------------------------------------------------------------------------
    timer.start("stage2b_coarse")

    # A heavily reduced version of each image is sufficient for estimating
    # the broad overlap. There is no reason to run expensive fine matching
    # on tens of thousands of pixels before we know where the two scenes
    # overlap.
    src_view, src_dec = io_utils.decimated_view(
        np.asarray(src_arr),
        config.MAX_COARSE_DIM
    )

    ref_view, ref_dec = io_utils.decimated_view(
        np.asarray(ref_arr),
        config.MAX_COARSE_DIM
    )

    # Combine preview/native stride with the additional coarse decimation
    # so we can translate coarse-image coordinates back to full resolution.
    src_total = (
        src_stride[0] * src_dec,
        src_stride[1] * src_dec
    )

    ref_total = (
        ref_stride[0] * ref_dec,
        ref_stride[1] * ref_dec
    )

    # Metadata gives us an initial estimate of where the two footprints
    # overlap. Restricting the reference search to this area is much more
    # reliable than trying to match two complete frames globally.
    overlap_window = overlap.compute_overlap_window(
        source_meta,
        reference_meta
    )

    cx0 = max(
        0,
        int(overlap_window["col0"] / ref_total[0])
    )
    cy0 = max(
        0,
        int(overlap_window["row0"] / ref_total[1])
    )

    cx1 = min(
        ref_view.shape[1],
        max(
            cx0 + 64,
            int(
                math.ceil(
                    overlap_window["col1"] / ref_total[0]
                )
            )
        )
    )

    cy1 = min(
        ref_view.shape[0],
        max(
            cy0 + 64,
            int(
                math.ceil(
                    overlap_window["row1"] / ref_total[1]
                )
            )
        )
    )

    # If the estimated overlap is too small to support useful matching,
    # fall back to the complete reference preview rather than creating an
    # unusably tiny search region.
    if (
        cx1 - cx0 < 64
        or cy1 - cy0 < 64
    ):
        cx0, cy0 = 0, 0
        cx1, cy1 = (
            ref_view.shape[1],
            ref_view.shape[0]
        )

    ref_roi_view = ref_view[
        cy0:cy1,
        cx0:cx1
    ]

    # Coarse matching only needs lightweight normalization/CLAHE. The full
    # sensor specific enhancement recipe is intentionally reserved for the
    # fine matching stage.
    src_coarse = enhance.apply_clahe(
        enhance.normalize_8bit(src_view)
    )

    ref_coarse = enhance.apply_clahe(
        enhance.normalize_8bit(ref_roi_view)
    )

    # The coarse matcher needs to know the physical scale difference between
    # the sensors. For OHRC -> NAC this can be a large difference, so this
    # factor is important for making the pyramid search meaningful.
    gsd_scale = (
        source_meta["pixel_resolution_m"]
        / reference_meta["pixel_resolution_m"]
    )

    best, _, _ = matching.match_coarse_sift(
        src_coarse,
        ref_coarse,
        gsd_scale,
        verbose=verbose
    )

    if best is not None:
        coarse_method = "sift_pyramid"

        ke = best["k_eff"]
        u = best["u"]
        Hc = best["Hc"]

        # Convert the coarse transformation back into full-image
        # coordinates. Each matrix accounts for one coordinate-system
        # change introduced during previewing, cropping, or pyramid scaling.
        Ssrc = np.diag([
            src_total[0],
            src_total[1],
            1.0
        ])

        Sref = np.diag([
            ref_total[0],
            ref_total[1],
            1.0
        ])

        Dk = np.diag([
            ke,
            ke,
            1.0
        ])

        Tc = np.array(
            [
                [1, 0, cx0],
                [0, 1, cy0],
                [0, 0, 1]
            ],
            dtype=float
        )

        Du_inv = np.diag([
            1.0 / u,
            1.0 / u,
            1.0
        ])

        H_full = (
            Sref
            @ Tc
            @ Du_inv
            @ Hc
            @ Dk
            @ np.linalg.inv(Ssrc)
        )

        if verbose:
            print(
                f"Coarse: sift_pyramid "
                f"k_eff={ke:.4f}, "
                f"{best['n_inliers']} inliers."
            )

        coarse_src_view = best["src_try"]
        coarse_ref_view = best["ref_try"]

    else:
        # Image based localization can fail when illumination, scale, or
        # viewing geometry differs too much. Metadata provides a useful
        # fallback because the sensor footprints still give us a coarse
        # spatial relationship.
        coarse_method = "metadata_corners"

        if verbose:
            print(
                "WARNING: image coarse match failed at all levels; "
                "falling back to metadata-corner homography."
            )

        H_full = geometry_fit.metadata_corner_homography(
            source_meta,
            reference_meta
        )

        coarse_src_view = src_coarse
        coarse_ref_view = ref_coarse

    if verbose:
        print(
            f"Coarse method: {coarse_method}\n"
            f"H_full:\n{H_full}"
        )

    timer.stop("stage2b_coarse")

    # -------------------------------------------------------------------------
    # Stage 2C: extract the likely overlap ROI
    # -------------------------------------------------------------------------
    timer.start("stage2c_roi")

    sh, sw = (
        int(source_meta["shape"][0]),
        int(source_meta["shape"][1])
    )

    rh, rw = (
        int(reference_meta["shape"][0]),
        int(reference_meta["shape"][1])
    )

    # Project the source footprint into the reference image using the
    # coarse transformation. This tells us where to look for the actual
    # source/reference overlap.
    ref_proj = _project_corners(
        H_full,
        sw,
        sh
    )

    ref_bbox = _clamp_bbox(
        float(ref_proj[:, 0].min()),
        float(ref_proj[:, 1].min()),
        float(ref_proj[:, 0].max()),
        float(ref_proj[:, 1].max()),
        rw,
        rh,
        pad=250
    )

    if (
        ref_bbox[2] - ref_bbox[0] < 100
        or ref_bbox[3] - ref_bbox[1] < 100
    ):
        raise RuntimeError(
            "Projected overlap is too small; "
            "coarse transform is probably wrong."
        )

    try:
        H_inv = np.linalg.inv(H_full)
    except np.linalg.LinAlgError:
        raise RuntimeError(
            "Coarse homography is singular."
        )

    x0, y0, x1, y1 = ref_bbox

    # Take the reference ROI corners and map them backwards into the source
    # image. This gives us the corresponding source-side bounding region.
    rc = np.array(
        [
            [x0, y0],
            [x1 - 1, y0],
            [x1 - 1, y1 - 1],
            [x0, y1 - 1]
        ],
        dtype=np.float32
    )

    sc = cv2.perspectiveTransform(
        rc.reshape(-1, 1, 2),
        H_inv
    ).reshape(-1, 2)

    src_bbox = _clamp_bbox(
        float(sc[:, 0].min()),
        float(sc[:, 1].min()),
        float(sc[:, 0].max()),
        float(sc[:, 1].max()),
        sw,
        sh,
        pad=150
    )

    # Extract only the regions needed for fine registration. This is
    # especially important for OHRC, where the complete source product
    # can be enormous.
    src_roi, src_origin, _ = _read_roi(
        np.asarray(src_arr),
        (src_stride[0], src_stride[1]),
        src_bbox
    )

    ref_roi, ref_origin, _ = _read_roi(
        np.asarray(ref_arr),
        (ref_stride[0], ref_stride[1]),
        ref_bbox
    )

    roi_src_origin = (
        src_origin[0],
        src_origin[1]
    )

    roi_src_scale = (
        src_stride[0],
        src_stride[1]
    )

    roi_ref_origin = (
        ref_origin[0],
        ref_origin[1]
    )

    roi_ref_scale = (
        ref_stride[0],
        ref_stride[1]
    )

    if verbose:
        print(
            f"Reference ROI (full): {ref_bbox} | "
            f"Source ROI (full): {src_bbox}"
        )

        print(
            f"ROI arrays: source {src_roi.shape} "
            f"ref {ref_roi.shape}"
        )

    timer.stop("stage2c_roi")

    # The coarse previews are no longer needed after ROI extraction.
    # Releasing them here is important when working with large products.
    io_utils.free_arrays(
        src_view,
        ref_view
    )

    # -------------------------------------------------------------------------
    # Stage 3: common physical scale
    # -------------------------------------------------------------------------
    timer.start("stage3_resample")

    (
        source_common_raw,
        reference_common_raw,
        fine_f,
        S_inv,
        T_ref,
        resample_info
    ) = resample.resample_roi_to_ref_scale(
        src_roi,
        ref_roi,
        roi_src_origin,
        roi_src_scale,
        roi_ref_origin,
        roi_ref_scale,
        source_meta,
        reference_meta
    )

    # Convert the full-resolution coarse homography into the local
    # coordinate system used by the resampled ROIs.
    H_common = (
        T_ref
        @ H_full
        @ S_inv
    )

    if verbose:
        print(
            f"Stage 3 resample: {resample_info}"
        )

    timer.stop("stage3_resample")

    # -------------------------------------------------------------------------
    # Stage 4: normalize image values
    # -------------------------------------------------------------------------
    timer.start("stage4_normalize")

    # The original sensor products may have different numerical ranges.
    # Convert both sides into a common 8-bit representation before applying
    # the feature-enhancement recipe.
    source_8bit = enhance.normalize_8bit(
        source_common_raw
    )

    reference_8bit = enhance.normalize_8bit(
        reference_common_raw
    )

    timer.stop("stage4_normalize")

    # -------------------------------------------------------------------------
    # Stage 5: sensor-specific enhancement
    # -------------------------------------------------------------------------
    timer.start("stage5_enhance")

    # Enhancement is kept separate from geometric registration so that
    # feature extraction can work on images optimized for correspondence
    # finding without altering the underlying coordinate system.
    (
        enhanced_source,
        enhanced_reference,
        enhancement_stages
    ) = enhance.enhance(
        sensor,
        source_8bit,
        reference_8bit,
        recipe=enhance_recipe
    )

    timer.stop("stage5_enhance")

    # -------------------------------------------------------------------------
    # Stage 6: coarse warp + fine correspondence search
    # -------------------------------------------------------------------------
    timer.start("stage6_matching")

    # Apply the coarse transformation first. Fine matching then operates
    # on images that are already approximately aligned rather than trying
    # to solve the entire geometric difference from scratch.
    coarse_warp = cv2.warpPerspective(
        enhanced_source,
        H_common,
        (
            enhanced_reference.shape[1],
            enhanced_reference.shape[0]
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # NCC is only a sanity check here. It tells us whether the coarse warp
    # looks broadly consistent with the reference before spending time on
    # fine correspondence extraction.
    coarse_ncc = geometry_fit.coarse_ncc(
        enhanced_reference,
        coarse_warp,
        coarse_warp > 0
    )

    if verbose:
        print(
            f"Coarse-warp check: NCC={coarse_ncc:.3f} "
            f"(expect >{config.COARSE_NCC_WARN} "
            f"if coarse ROI is right)"
        )

    # SuperGlue is the preferred matcher. SIFT can be used as a fallback
    # when the primary matcher fails or is unavailable.
    (
        mkpts_src,
        mkpts_ref,
        match_conf,
        matcher_used
    ) = matching.match_fine(
        matcher,
        coarse_warp,
        enhanced_reference,
        use_fallback=use_fallback,
        verbose=verbose
    )

    if verbose:
        print(
            f"Stage 6: {len(mkpts_src)} raw matches "
            f"(matcher={matcher_used})."
        )

    timer.stop("stage6_matching")

    # -------------------------------------------------------------------------
    # Stage 7: geometric fitting + spatial selection + refinement
    # -------------------------------------------------------------------------
    timer.start("stage7_geometry_fit")

    # Estimate the geometric transformation and reject correspondences
    # that cannot be explained by the fitted model.
    fit = geometry_fit.fit_and_evaluate(
        mkpts_src,
        mkpts_ref,
        use_piecewise=use_piecewise
    )

    H_final = None
    final_src = None
    final_ref = None

    rmse = {
        "rmse_x": None,
        "rmse_y": None,
        "rmse_total": None
    }

    uniformity = None
    median_err = None
    refine_accepted = False

    if (
        fit["registration_accepted"]
        and fit["mode"] == "global"
        and fit["H"] is not None
    ):
        # A geometrically valid match set can still be badly distributed.
        # Select points with better spatial coverage before the final
        # refinement step.
        (
            Huni,
            fs,
            fr,
            _,
            _,
            _
        ) = geometry_fit.fit_uniform(
            mkpts_src,
            mkpts_ref,
            match_conf,
            enhanced_reference.shape
        )

        if (
            Huni is not None
            and len(fs) >= 4
        ):
            # Refine the transformation using the spatially selected points.
            H2, fs2, fr2, _, _, acc = (
                geometry_fit.refine_and_keep(
                    coarse_warp,
                    enhanced_reference,
                    fs,
                    fr,
                    Huni
                )
            )

            H_final = H2
            final_src = fs2
            final_ref = fr2
            refine_accepted = acc

            # Reprojection residuals provide the main geometric accuracy
            # measure. These are more meaningful for registration than a
            # simple image similarity score.
            rx = (
                cv2.perspectiveTransform(
                    final_src.reshape(-1, 1, 2),
                    H_final
                ).reshape(-1, 2)
                - final_ref
            )

            rmse = {
                "rmse_x": float(
                    np.sqrt(np.mean(rx[:, 0] ** 2))
                ),
                "rmse_y": float(
                    np.sqrt(np.mean(rx[:, 1] ** 2))
                ),
                "rmse_total": float(
                    np.sqrt(
                        np.mean(
                            np.sum(rx ** 2, axis=1)
                        )
                    )
                )
            }

            median_err = float(
                np.median(
                    np.linalg.norm(rx, axis=1)
                )
            )

            uniformity = (
                geometry_fit.uniformity_coverage(
                    final_ref,
                    enhanced_reference.shape
                )
            )

            fit["uniform_applied"] = True
            fit["refine_accepted"] = refine_accepted
            fit["H"] = H_final
            fit["n_inliers"] = int(
                len(final_src)
            )
            fit["inlier_ratio"] = float(
                len(final_src)
                / max(len(mkpts_src), 1)
            )

            # Re-run the final quality gate after spatial selection and
            # refinement, because the accepted correspondence set has now
            # changed.
            fit["registration_accepted"] = (
                geometry_fit.quality_gate(
                    fit["n_inliers"],
                    fit["inlier_ratio"]
                )
            )

        else:
            # If uniform selection/refinement cannot produce a valid model,
            # retain the original global fit rather than discarding an
            # otherwise usable registration.
            H_final = fit["H"]

            final_src = mkpts_src[
                fit["inlier_mask"]
            ]

            final_ref = mkpts_ref[
                fit["inlier_mask"]
            ]

            rmse = evaluate.compute_rmse(
                mkpts_src,
                mkpts_ref,
                fit["H"],
                fit["inlier_mask"]
            )

            uniformity = (
                geometry_fit.uniformity_coverage(
                    final_ref,
                    enhanced_reference.shape
                )
            )

            median_err = (
                geometry_fit.median_reproj_error(
                    mkpts_src,
                    mkpts_ref,
                    fit["H"],
                    fit["inlier_mask"]
                )
            )

    elif fit["registration_accepted"]:
        # Legacy piecewise path. Each segment has its own model, so the
        # global uniform-selection/refinement step is not applied here.
        seg0 = fit["segments"][0]

        rmse = evaluate.compute_rmse(
            mkpts_src,
            mkpts_ref,
            seg0["H"],
            fit["inlier_mask"]
        )

        uniformity = (
            geometry_fit.uniformity_coverage(
                mkpts_ref[fit["inlier_mask"]],
                enhanced_reference.shape
            )
        )

        median_err = (
            geometry_fit.median_reproj_error(
                mkpts_src,
                mkpts_ref,
                seg0["H"],
                fit["inlier_mask"]
            )
        )

    if verbose:
        print(
            f"Stage 7: mode={fit['mode']}, "
            f"n_inliers={fit['n_inliers']}, "
            f"inlier_ratio={fit['inlier_ratio']:.3f}, "
            f"accepted={fit['registration_accepted']}, "
            f"refine={refine_accepted}"
        )

    timer.stop("stage7_geometry_fit")

    # -------------------------------------------------------------------------
    # Stage 8–10: final product and evaluation
    # -------------------------------------------------------------------------
    warped = None
    coordinate_anchor = None
    psnr_value = None
    ssim_value = None

    if fit["registration_accepted"]:

        # ---------------------------------------------------------------------
        # Stage 8: warp
        # ---------------------------------------------------------------------
        timer.start("stage8_warp")

        if fit["mode"] == "piecewise":
            warped = warp.warp_source(
                source_8bit,
                reference_8bit.shape,
                fit
            )
        else:
            warped = warp.tiled_warp(
                source_8bit,
                reference_8bit.shape,
                H_final if H_final is not None else fit["H"]
            )

        timer.stop("stage8_warp")

        # ---------------------------------------------------------------------
        # Stage 9: coordinate re-anchoring
        # ---------------------------------------------------------------------
        timer.start("stage9_reanchor")

        # Record how the final image relates to the reference frame. This
        # keeps the spatial provenance alongside the image output.
        coordinate_anchor = warp.reanchor_to_reference(
            reference_meta,
            overlap_window
        )

        timer.stop("stage9_reanchor")

        # ---------------------------------------------------------------------
        # Stage 10: evaluation
        # ---------------------------------------------------------------------
        timer.start("stage10_evaluate")

        metrics = evaluate.Metrics(
            reference_8bit,
            warped
        )

        psnr_value = metrics.psnr()
        ssim_value = metrics.ssim()

        timer.stop("stage10_evaluate")

    elif verbose:
        # Do not generate a misleading registered product when the geometry
        # failed the quality gate.
        print(
            "Stage 8-10 skipped: registration was not accepted "
            "by the quality gate."
        )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    # Convert the collected information into a compact, serializable report.
    # Large image arrays remain in the returned result for visualization but
    # are not unnecessarily duplicated inside the report itself.
    report = evaluate.build_report(
        pair_name=pair_name,
        sensor=sensor,
        geometry_summary=geometry_check,
        fit_result=fit,
        rmse=rmse,
        psnr=psnr_value,
        ssim=ssim_value,
        timer=timer,
        uniformity=uniformity,
        median_err=median_err,
        matcher_used=matcher_used,
        coarse_method=coarse_method,
        source_band=src_band,
        source_pixels=(
            "preview_png_rapid_look"
            if use_previews_only
            else "full_img"
        ),
        reference_pixels=(
            "preview_png_rapid_look"
            if use_previews_only
            else "full_img"
        ),
        coarse_ncc=coarse_ncc,
    )

    # Store the ROI and scale information explicitly so the final report
    # can be interpreted without having to reconstruct intermediate state.
    report["reference_roi_full_px"] = [
        int(v)
        for v in ref_bbox
    ]

    report["source_roi_full_px"] = [
        int(v)
        for v in src_bbox
    ]

    report["fine_resample_f"] = float(
        fine_f
    )

    # Return both the final result and the intermediate products. Keeping
    # these arrays available is extremely useful when debugging a failed
    # registration: we can inspect the coarse warp, enhancement stages,
    # match locations, and final geometry without rerunning every stage.
    return {
        "source_meta": source_meta,
        "reference_meta": reference_meta,

        "geometry_check": geometry_check,
        "overlap_window": overlap_window,

        "coarse_method": coarse_method,
        "H_full": H_full,
        "H_common": H_common,

        "source_roi": src_bbox,
        "reference_roi": ref_bbox,

        "source_array": source_8bit,
        "reference_array": reference_8bit,

        "resample_info": resample_info,

        "enhanced_source": enhanced_source,
        "enhanced_reference": enhanced_reference,
        "enhancement_stages": enhancement_stages,

        "coarse_warp": coarse_warp,
        "coarse_ncc": coarse_ncc,

        "mkpts_src": mkpts_src,
        "mkpts_ref": mkpts_ref,
        "match_conf": match_conf,
        "matcher_used": matcher_used,

        "final_src": final_src,
        "final_ref": final_ref,

        "fit": fit,
        "warped": warped,
        "coordinate_anchor": coordinate_anchor,

        "report": report,
    }