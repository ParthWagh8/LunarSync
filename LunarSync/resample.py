"""
Stage 3 of the registration pipeline.

The purpose of this stage is to bring the source and reference images to
a comparable ground sampling distance (GSD) before feature matching.

For example, an OHRC image can be around 0.25 m/pixel while an LRO NAC
image may be around 1.6 m/pixel. Without scale normalization, the same
lunar feature can occupy very different numbers of pixels in the two
images, making correspondence finding much harder.

The interpolation method is selected using the source image's Sun
elevation. The underlying papers suggest that bicubic interpolation can
preserve useful fine detail in well-lit scenes, while its larger
neighbourhood can also emphasize shadow-related artefacts in low-sun-angle
imagery.

This interpolation rule is therefore a synthesis of the reported
experimental results rather than a direct rule stated by either paper.
"""

import cv2

import LunarSync.config as config


def choose_interpolation(meta):
    """
    Select the interpolation method based on Sun elevation.

    Higher Sun elevation generally means better illumination and more
    visible fine-scale surface detail, where bicubic interpolation can
    be useful.

    For low or unknown Sun elevation we use bilinear interpolation as the
    safer default, avoiding unnecessary sharpening of strong shadow
    boundaries.

    Returns:
        (OpenCV interpolation flag, human-readable description)
    """

    elev = meta.get("sun_elevation_deg")

    # If the Sun elevation is high enough, preserve more local detail.
    # Otherwise use the more conservative bilinear interpolation.
    if (
        elev is not None
        and elev >= config.BICUBIC_SUN_ELEVATION_DEG
    ):
        return (
            cv2.INTER_CUBIC,
            "bicubic (high sun elevation)"
        )

    return (
        cv2.INTER_LINEAR,
        "bilinear (low or unknown sun elevation)"
    )


def resample_to_gsd(
    array,
    source_gsd_m,
    target_gsd_m,
    interpolation
):
    """
    Resize an image so that its effective GSD matches the requested
    target GSD.

    The physical relationship is:

        scale = source_GSD / target_GSD

    For example, converting 0.25 m/pixel data to 1.0 m/pixel data gives:

        scale = 0.25 / 1.0 = 0.25

    meaning the image becomes roughly one quarter of its original
    dimensions.

    Returns:
        resized_array
        scale_factor
    """

    if source_gsd_m is None or target_gsd_m is None:
        raise ValueError(
            "Both source and target GSD must be known to resample."
        )

    # This is a physical-scale conversion, not just an arbitrary image
    # resize. The ratio tells us how many output pixels are needed to
    # represent the same ground distance.
    scale = source_gsd_m / target_gsd_m

    # Avoid an unnecessary OpenCV resize when the two images already have
    # effectively identical ground sampling distances.
    if abs(scale - 1.0) < 1e-6:
        return array, 1.0

    h, w = array.shape[:2]

    new_w = max(
        1,
        int(round(w * scale))
    )
    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        array,
        (new_w, new_h),
        interpolation=interpolation
    )

    return resized, scale


def rescale_to_common_gsd(
    source_array,
    source_meta,
    reference_array,
    reference_meta
):
    """
    Bring the source and reference images to a common physical scale.

    The finer-resolution image is reduced to the coarser image's GSD.
    This avoids unnecessarily upsampling the lower-resolution image and
    creating artificial detail that was never present in the original
    data.

    Example:
        OHRC  ~0.25 m/pixel
        NAC   ~1.6  m/pixel

    In this case, OHRC is downsampled to approximately the NAC scale.

    Returns:
        source_out
        reference_out
        info_dict
    """

    source_gsd = source_meta.get(
        "pixel_resolution_m"
    )
    reference_gsd = reference_meta.get(
        "pixel_resolution_m"
    )

    if source_gsd is None or reference_gsd is None:
        raise ValueError(
            "Both source and reference must have "
            "pixel_resolution_m in their metadata."
        )

    # Use the source image's illumination conditions when choosing the
    # interpolation kernel. This keeps the resampling decision tied to the
    # actual imaging conditions rather than to an arbitrary fixed kernel.
    interpolation, interp_label = choose_interpolation(
        source_meta
    )

    if source_gsd < reference_gsd:
        # The source contains finer spatial detail, so reduce it to the
        # reference scale instead of enlarging the reference.
        resized_source, scale = resample_to_gsd(
            source_array,
            source_gsd,
            reference_gsd,
            interpolation
        )

        info = {
            "resized": "source",
            "scale_factor": scale,
            "interpolation": interp_label,
            "source_gsd_m": source_gsd,
            "reference_gsd_m": reference_gsd,
        }

        return (
            resized_source,
            reference_array,
            info
        )

    elif reference_gsd < source_gsd:
        # The reference is the finer-resolution image. The same principle
        # applies in the opposite direction: reduce the reference to the
        # source's physical scale.
        resized_reference, scale = resample_to_gsd(
            reference_array,
            reference_gsd,
            source_gsd,
            interpolation
        )

        info = {
            "resized": "reference",
            "scale_factor": scale,
            "interpolation": interp_label,
            "source_gsd_m": source_gsd,
            "reference_gsd_m": reference_gsd,
        }

        return (
            source_array,
            resized_reference,
            info
        )

    else:
        # Both sensors already describe roughly the same ground distance
        # per pixel, so resizing would only introduce unnecessary
        # interpolation.
        info = {
            "resized": "none",
            "scale_factor": 1.0,
            "interpolation": interp_label,
            "source_gsd_m": source_gsd,
            "reference_gsd_m": reference_gsd,
        }

        return (
            source_array,
            reference_array,
            info
        )


def resample_roi_to_ref_scale(
    source_roi,
    reference_roi,
    roi_src_origin_raw,
    roi_src_scale,
    roi_ref_origin_raw,
    roi_ref_scale,
    source_meta,
    reference_meta
):
    """
    Resample an already-extracted source ROI to the physical scale of
    the reference ROI.

    This function is slightly more complicated than
    rescale_to_common_gsd() because the ROIs do not necessarily represent
    raw detector pixels.

    For example, a preview image might have been created by taking every
    8th raw pixel. In that case, one preview pixel represents:

        sensor_GSD * 8

    metres on the ground.

    We therefore keep track of both:
        1. where the ROI came from in the original image
        2. how many raw pixels each ROI pixel represents

    This prevents the coordinate transforms from accidentally treating
    preview pixels as if they were raw pixels.

    Returns:
        source_common
        reference_common
        f
        S_inv
        T_ref
        info
    """

    import numpy as np

    source_gsd = source_meta.get(
        "pixel_resolution_m"
    )
    reference_gsd = reference_meta.get(
        "pixel_resolution_m"
    )

    if source_gsd is None or reference_gsd is None:
        raise ValueError(
            "Both source and reference must have "
            "pixel_resolution_m in their metadata."
        )

    # Convert the ROI sampling intervals into actual ground sampling
    # distances. roi_*_scale accounts for any preview/downsampling that
    # happened before this function received the ROI.
    sgsx = source_gsd * float(roi_src_scale[0])
    sgsy = source_gsd * float(roi_src_scale[1])

    rgsx = reference_gsd * float(roi_ref_scale[0])
    rgsy = reference_gsd * float(roi_ref_scale[1])

    # Use the average of the horizontal and vertical ground scales.
    # The current pipeline assumes approximately square pixels, but keeping
    # x/y values separate above makes the coordinate bookkeeping explicit.
    f = (
        ((sgsx + sgsy) / 2)
        / ((rgsx + rgsy) / 2)
    )

    if f <= 0:
        raise ValueError(
            "Invalid ROI ground scales."
        )

    # Start with the same Sun-angle-aware interpolation rule used by the
    # full-image path.
    interpolation, interp_label = choose_interpolation(
        source_meta
    )

    if f < 1.0:
        # When shrinking an image, area interpolation is preferable because
        # it reduces aliasing while combining multiple source pixels into
        # one output pixel.
        interpolation = cv2.INTER_AREA
        interp_label = "area (shrinking)"

    h, w = source_roi.shape[:2]

    new_w = max(
        1,
        int(round(w * f))
    )
    new_h = max(
        1,
        int(round(h * f))
    )

    if abs(f - 1.0) < 1e-6:
        # Already at the required physical scale. Avoid a needless resize.
        source_common = source_roi
    else:
        source_common = cv2.resize(
            source_roi,
            (new_w, new_h),
            interpolation=interpolation
        )

    # The reference ROI is already our target scale, so it remains unchanged.
    reference_common = reference_roi

    # Unpack the sampling factors and original ROI origins. These values are
    # needed to convert feature coordinates between:
    #
    #     common-image pixels
    #          <-> raw source pixels
    #          <-> reference ROI pixels
    #
    # Keeping this transformation explicit is important because a match
    # found on a downsampled ROI cannot be interpreted directly as a raw
    # detector coordinate.
    ssx = float(roi_src_scale[0])
    ssy = float(roi_src_scale[1])

    osx = float(roi_src_origin_raw[0])
    osy = float(roi_src_origin_raw[1])

    rsx = float(roi_ref_scale[0])
    rsy = float(roi_ref_scale[1])

    orx = float(roi_ref_origin_raw[0])
    ory = float(roi_ref_origin_raw[1])

    # Map coordinates from the common source image back to raw source
    # coordinates.
    #
    # The division by f accounts for the source ROI's rescaling, while the
    # origin terms put the coordinates back into the full raw image frame.
    S_inv = np.array(
        [
            [ssx / f, 0, osx],
            [0, ssy / f, osy],
            [0, 0, 1],
        ],
        dtype=np.float64
    )

    # Convert reference ROI coordinates into the full reference frame.
    # Unlike the source side, the reference ROI is not rescaled here.
    T_ref = np.array(
        [
            [1 / rsx, 0, -orx / rsx],
            [0, 1 / rsy, -ory / rsy],
            [0, 0, 1],
        ],
        dtype=np.float64
    )

    info = {
        "resample_factor_f": f,
        "interpolation": interp_label,
        "source_roi_ground_gsd_m": (
            (sgsx + sgsy) / 2
        ),
        "reference_roi_ground_gsd_m": (
            (rgsx + rgsy) / 2
        ),
        "source_gsd_m": source_gsd,
        "reference_gsd_m": reference_gsd,
    }

    return (
        source_common,
        reference_common,
        f,
        S_inv,
        T_ref,
        info
    )