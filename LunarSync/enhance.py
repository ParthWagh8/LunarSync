"""
Stages 4-5 of the registration pipeline.

This module handles the intensity normalization and sensor-specific image
enhancement used before feature matching.

The pipeline supports two enhancement paths:

    OHRC / TMC-2 -> NAC
        CLAHE -> Invert -> Dilate -> PCA

    IIRS -> NAC
        Histogram Match -> Shadow Normalize -> Log Transform

Each enhancement function returns:
    1. The final enhanced source image
    2. The final enhanced reference image
    3. A dictionary containing all intermediate stages

The intermediate images are mainly useful for debugging and notebook
visualization. The matching pipeline only needs the final enhanced images.
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage import exposure

import LunarSync.config as config


# ---------------------------------------------------------------------------
# Shared image-processing building blocks
# ---------------------------------------------------------------------------

def normalize_8bit(
    array,
    low_pct=config.NORMALIZE_LOW_PCT,
    high_pct=config.NORMALIZE_HIGH_PCT
):
    """
    Stretch a numeric image into the full 8-bit range.

    Percentile-based normalization is used instead of directly taking the
    minimum and maximum. This makes the stretch less sensitive to a few
    extreme pixels or noisy values in large remote-sensing images.
    """

    a = array.astype(np.float32)

    # Ignore NaN/Inf values when estimating the useful intensity range.
    finite = a[np.isfinite(a)]

    # If the image contains no valid pixels, return a blank uint8 image
    # rather than allowing the percentile calculation to fail.
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, [low_pct, high_pct])

    # A nearly constant image cannot be stretched normally. Fall back to
    # the actual data range while adding a tiny epsilon to avoid division
    # by zero.
    if hi <= lo:
        lo, hi = finite.min(), finite.max() + 1e-6

    # Map the selected percentile range to [0, 255] and clip everything
    # outside it. Values below/above the chosen percentiles therefore
    # become 0/255 respectively.
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0

    return a.astype(np.uint8)


def apply_clahe(
    gray8,
    clip_limit=config.CLAHE_CLIP_LIMIT,
    tile_grid=config.CLAHE_TILE_GRID
):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE improves local contrast rather than applying one global histogram
    operation. This can make small crater edges, ridges, and other local
    structures easier for feature detectors to distinguish.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )

    return clahe.apply(gray8)


def invert(gray8):
    """
    Invert an 8-bit grayscale image.

    Dark pixels become bright and bright pixels become dark. This is included
    because the OHRC/TMC-2 enhancement recipe uses intensity inversion as an
    intermediate step.
    """

    return 255 - gray8


def dilate(
    gray8,
    kernel_size=config.DILATE_KERNEL_SIZE,
    iterations=config.DILATE_ITERATIONS
):
    """
    Apply morphological dilation using an elliptical structuring element.

    Dilation expands bright structures in the inverted image. In this
    pipeline it is intended to strengthen local structural patterns before
    the PCA-based enhancement stage.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    return cv2.dilate(
        gray8,
        kernel,
        iterations=iterations
    )


def pca_enhance(
    gray8,
    variance_retained=config.PCA_VARIANCE_RETAINED
):
    """
    Enhance the image using a row-wise PCA reconstruction.

    The image is treated as a matrix where each row is a feature vector.
    PCA keeps enough components to retain the requested fraction of the
    variance, then reconstructs the image from those components.

    The reconstruction is normalized again to uint8 so that the output
    remains compatible with OpenCV-based feature extraction.

    Important:
        This is a reasonable implementation of the described PCA stage,
        but it is not an exact reproduction of a published formula because
        the source paper does not specify the complete implementation.
    """

    a = gray8.astype(np.float32)

    # Let PCA automatically determine the number of components needed to
    # retain the requested amount of variance.
    pca = PCA(
        n_components=variance_retained,
        svd_solver="full"
    )

    # Project the image rows into the reduced PCA space.
    transformed = pca.fit_transform(a)

    # Reconstruct the image from the retained principal components.
    reconstructed = pca.inverse_transform(transformed)

    # Convert the reconstruction back into a normal 8-bit image.
    return normalize_8bit(reconstructed)


def histogram_match(source8, reference8):
    """
    Match the source intensity distribution to the reference distribution.

    This is useful for multi-sensor imagery where the same lunar terrain can
    have noticeably different brightness distributions because of sensor
    characteristics or acquisition conditions.
    """

    matched = exposure.match_histograms(
        source8,
        reference8
    )

    return matched.astype(np.uint8)


def shadow_normalize(
    gray8,
    shadow_percentile=config.SHADOW_PERCENTILE,
    gain=config.SHADOW_GAIN
):
    """
    Brighten the darker part of the image using an adaptive gamma.

    The darkest percentile of pixels is treated as the shadow region. Its
    mean brightness determines the gamma correction, allowing darker images
    to receive stronger brightening without applying a fixed correction to
    every scene.
    """

    # Identify the intensity threshold defining the shadow region.
    thresh = np.percentile(gray8, shadow_percentile)

    # Include pixels below the threshold while avoiding an excessively
    # restrictive threshold near zero.
    shadow_mask = gray8 <= max(thresh, 1)

    # Estimate the average brightness of the shadowed pixels. If no pixels
    # satisfy the mask, use 1.0 to keep the calculation well-defined.
    mean_shadow = (
        gray8[shadow_mask].mean()
        if shadow_mask.any()
        else 1.0
    )

    # Convert the shadow brightness into an adaptive gamma value. The gain
    # controls how strongly dark regions are lifted.
    gamma = 1.0 / max(
        gain * (mean_shadow / 255.0 + 0.15),
        0.3
    )

    # Keep the correction within a practical range so an unusual image does
    # not receive an excessively aggressive gamma transformation.
    gamma = float(np.clip(gamma, 0.4, 2.5))

    # Gamma correction is easier to express in normalized [0, 1] intensity.
    norm = gray8.astype(np.float32) / 255.0

    out = np.power(norm, gamma) * 255.0

    return out.astype(np.uint8)


def log_transform(gray8):
    """
    Apply a logarithmic intensity transformation.

    The logarithm expands differences among darker pixels while compressing
    high intensities. This can expose low-contrast terrain details that are
    otherwise difficult to distinguish in shadowed regions.
    """

    a = gray8.astype(np.float32)

    # Scale log1p output so that the maximum possible input value (255)
    # maps approximately back to 255.
    c = 255.0 / np.log1p(255.0)

    out = c * np.log1p(a)

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sensor-specific enhancement recipes
# ---------------------------------------------------------------------------

def enhance_ohrc_or_tmc2(source8, reference8):
    """
    Apply the OHRC/TMC-2 -> NAC enhancement recipe:

        CLAHE -> Invert -> Dilate -> PCA

    The same sequence is applied independently to both the source image and
    the NAC reference tile. Keeping both sides in a comparable enhanced
    representation helps the subsequent feature matcher.

    Returns:
        enhanced_source
        enhanced_reference
        stages

    `stages` contains every intermediate image so the notebook can display
    how the enhancement pipeline changes the imagery step by step.
    """

    stages = {}
    out = {}

    # Run the exact same enhancement sequence on both sides of the
    # registration pair. This keeps the processing symmetric.
    for side, img in [
        ("source", source8),
        ("reference", reference8)
    ]:

        # First increase local contrast.
        clahe_img = apply_clahe(img)

        # Invert the enhanced image as required by the recipe.
        invert_img = invert(clahe_img)

        # Strengthen bright structural patterns after inversion.
        dilate_img = dilate(invert_img)

        # Reduce/reconstruct the dominant image structure using PCA.
        pca_img = pca_enhance(dilate_img)

        # Save every stage for diagnostic visualization.
        stages[f"{side}_original"] = img
        stages[f"{side}_clahe"] = clahe_img
        stages[f"{side}_invert"] = invert_img
        stages[f"{side}_dilate"] = dilate_img
        stages[f"{side}_pca"] = pca_img

        # The PCA result is the image actually passed to matching.
        out[side] = pca_img

    return (
        out["source"],
        out["reference"],
        stages
    )


def enhance_iirs(source8, reference8):
    """
    Apply the IIRS -> NAC enhancement recipe:

        Histogram Match -> Shadow Normalize -> Log Transform

    Histogram matching is applied first to the IIRS source so that its
    intensity distribution is brought closer to the NAC reference.

    The shadow and logarithmic transformations are then applied to both
    images.

    Returns:
        enhanced_source
        enhanced_reference
        stages
    """

    stages = {
        "source_original": source8,
        "reference_original": reference8
    }

    # Bring the IIRS source intensity distribution closer to the NAC
    # reference before performing the remaining enhancement steps.
    matched = histogram_match(
        source8,
        reference8
    )

    stages["source_histogram_matched"] = matched

    # Normalize darker source regions.
    source_shadow = shadow_normalize(matched)

    stages["source_shadow_normalized"] = source_shadow

    # Compress the remaining high intensities while preserving more detail
    # in the darker terrain.
    source_log = log_transform(source_shadow)

    stages["source_log_transform"] = source_log

    # The NAC reference does not need histogram matching because it is the
    # reference distribution. It still receives the shadow and log stages
    # so that both final images are processed consistently.
    reference_shadow = shadow_normalize(reference8)

    stages["reference_shadow_normalized"] = reference_shadow

    reference_log = log_transform(reference_shadow)

    stages["reference_log_transform"] = reference_log

    return (
        source_log,
        reference_log,
        stages
    )


# ---------------------------------------------------------------------------
# Lightweight enhancement path
# ---------------------------------------------------------------------------

def enhance_light(source8, reference8):
    """
    Apply a lightweight enhancement:

        CLAHE only

    This is a practical alternative to the full Paper-B recipe.

    Full-image PCA can be computationally expensive for very large lunar
    images, while inversion and dilation may also alter the appearance of
    crater boundaries. The lightweight path therefore provides a cheaper
    and more conservative preprocessing option for experimentation and
    Colab-scale runs.

    This path should not be interpreted as the full Paper-B recipe.
    """

    # Apply CLAHE once to each side. The result is deliberately kept simple
    # so this path remains inexpensive compared with PCA-based processing.
    source_clahe = apply_clahe(source8)
    reference_clahe = apply_clahe(reference8)

    stages = {
        "source_original": source8,
        "source_clahe": source_clahe,
        "reference_original": reference8,
        "reference_clahe": reference_clahe,
    }

    return (
        source_clahe,
        reference_clahe,
        stages
    )


# ---------------------------------------------------------------------------
# Enhancement dispatcher
# ---------------------------------------------------------------------------

def enhance(
    sensor,
    source8,
    reference8,
    recipe="paper"
):
    """
    Select the appropriate enhancement pipeline.

    Parameters
    ----------
    sensor : str
        Expected values are:
            "ohrc"
            "tmc"
            "iirs"

    source8 : np.ndarray
        8-bit source/moving image.

    reference8 : np.ndarray
        8-bit reference/fixed image.

    recipe : str
        "paper"  -> use the sensor-specific enhancement recipe.
        "light"  -> use CLAHE-only preprocessing.

    Returns
    -------
    tuple
        (enhanced_source, enhanced_reference, stages)
    """

    # The lightweight path is sensor-independent and takes priority when
    # explicitly requested.
    if recipe == "light":
        return enhance_light(
            source8,
            reference8
        )

    # OHRC and TMC-2 share the same enhancement recipe in this pipeline.
    if sensor in ("ohrc", "tmc"):
        return enhance_ohrc_or_tmc2(
            source8,
            reference8
        )

    # IIRS uses its own modality-specific preprocessing sequence.
    if sensor == "iirs":
        return enhance_iirs(
            source8,
            reference8
        )

    # Fail early if the caller provides an unsupported sensor name.
    raise ValueError(
        f"Unknown sensor '{sensor}' -- "
        "expected one of: ohrc, tmc, iirs"
    )