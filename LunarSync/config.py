"""
Central configuration for the lunar image registration pipeline.

All tunable parameters, thresholds, and output paths are kept in this file.
The rest of the pipeline should import these values instead of defining its
own copies of the same constants.

This makes experiments easier to reproduce: if a threshold, image size, or
output location needs to change, it can be changed here without modifying
the processing code itself.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory where processed previews, intermediate outputs, and reports
# are stored. This path is written for a Google Colab + Google Drive setup.
PROCESSED_ROOT = Path("/content/drive/MyDrive/processed")

# Output folders used to keep the results from different sensors and the
# evaluation reports separated.
OUTPUT_SUBDIRS = [
    "ohrc",
    "tmc",
    "nac",
    "reports",
]


# ---------------------------------------------------------------------------
# Planetary constants
# ---------------------------------------------------------------------------

# Mean lunar radius in kilometres.
#
# This is specifically the Moon's radius, not Earth's. overlap.py uses it
# when converting geographic coordinate differences into approximate lunar
# surface distances with the Haversine formula.
MOON_RADIUS_KM = 1737.4


# ---------------------------------------------------------------------------
# Metadata / illumination risk (Stage 0)
# ---------------------------------------------------------------------------

# Images acquired with a very low Sun elevation tend to contain larger
# shadowed regions and weaker visible surface texture.
#
# This threshold is used to flag potentially difficult image pairs. It is
# informational only and does not automatically reject the pair.
LOW_SUN_ELEVATION_DEG = 15.0


# ---------------------------------------------------------------------------
# Resampling (Stage 3)
# ---------------------------------------------------------------------------

# Select the interpolation method according to the Sun elevation.
#
# Higher Sun elevation generally provides stronger visible texture, where
# bicubic interpolation can be used. For low-Sun-angle imagery, the pipeline
# prefers bilinear interpolation because Paper A reports that bicubic
# interpolation can hurt matching performance in those conditions.
BICUBIC_SUN_ELEVATION_DEG = 20.0


# ---------------------------------------------------------------------------
# Enhancement (Stages 4-5)
# ---------------------------------------------------------------------------

# Percentile limits used by normalize_8bit() to map the useful intensity
# range into [0, 255]. Extreme outliers outside this range are clipped.
NORMALIZE_LOW_PCT = 0.5
NORMALIZE_HIGH_PCT = 99.5

# CLAHE controls how aggressively local contrast is enhanced.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Morphological dilation settings used by the OHRC/TMC-2 enhancement recipe.
DILATE_KERNEL_SIZE = 3
DILATE_ITERATIONS = 1

# Fraction of image variance that the PCA enhancement attempts to retain.
# A value of 0.95 means PCA keeps enough components to explain approximately
# 95% of the variance in the image.
PCA_VARIANCE_RETAINED = 0.95

# Parameters for the IIRS shadow-normalization stage.
#
# The darkest portion of the image is used to estimate shadow brightness,
# and SHADOW_GAIN controls the strength of the resulting gamma correction.
SHADOW_PERCENTILE = 15
SHADOW_GAIN = 1.6


# ---------------------------------------------------------------------------
# SuperGlue matching (Stage 6)
# ---------------------------------------------------------------------------

# Pre-trained SuperGlue weight set. "outdoor" is used because the matching
# problem involves large-scale natural surface structure rather than
# conventional indoor scenes.
SUPERGLUE_WEIGHTS = "outdoor"

# Maximum number of keypoints considered by the SuperPoint detector.
# Keeping this bounded prevents memory usage from growing too much on
# high-resolution lunar tiles.
MAX_KEYPOINTS = 4096

# Minimum SuperPoint keypoint detection confidence.
KEYPOINT_THRESHOLD = 0.003

# Non-maximum suppression radius used to prevent many nearly identical
# keypoints from being selected in the same local area.
NMS_RADIUS = 4

# Number of Sinkhorn iterations used by SuperGlue's optimal-transport
# matching stage.
SINKHORN_ITERATIONS = 20

# Minimum SuperGlue match confidence accepted as a valid correspondence.
MATCH_THRESHOLD = 0.2

# Before SuperGlue processing, a tile is resized so that its longest side
# does not exceed this value. This keeps GPU/CPU memory usage manageable
# while preserving the overall tile structure.
RESIZE_LONG_SIDE = 1600

# Size of the source/reference tiles used during large-image matching.
TILE_PX = 2000

# Neighboring tiles overlap by this many pixels. The overlap reduces the
# chance that useful features close to a tile boundary are lost.
TILE_OVERLAP_PX = 200

# Very flat tiles are unlikely to contain useful feature information.
# Tiles whose standard deviation is below this threshold are skipped,
# reducing unnecessary SuperGlue computation on blank or heavily shadowed
# regions.
SKIP_UNIFORM_TILE_STD = 2.0


# ---------------------------------------------------------------------------
# SIFT fallback matching (Stage 6B)
#
# DoG scale-space + 128-dimensional gradient descriptors follow the
# conventional SIFT formulation (Lowe 2004), with OpenCV providing the
# implementation.
#
# Matching uses BFMatcher with the L2 distance and Lowe's ratio test.
# ---------------------------------------------------------------------------

# Minimum contrast required for a SIFT feature to be considered a stable
# Difference-of-Gaussians extremum.
SIFT_CONTRAST_THRESHOLD = 0.04

# Rejects keypoints whose local response is too edge-like. This helps avoid
# unstable features that are poorly localized along strong edges.
SIFT_EDGE_THRESHOLD = 10.0

# 0 means OpenCV is allowed to detect as many SIFT features as necessary.
# Tiling the image still provides an upper bound on how much data each
# matching operation processes.
SIFT_N_FEATURES = 0

# Number of layers sampled inside each SIFT octave.
SIFT_N_OCTAVE_LAYERS = 3

# Lowe's nearest-neighbour ratio threshold.
#
# A match is accepted only when the best descriptor match is sufficiently
# better than the second-best match. Lower values are stricter.
SIFT_LOWE_RATIO = 0.75

# Require the correspondence to be mutual: the source feature must select
# the reference feature as its nearest match and vice versa.
SIFT_CROSS_CHECK = True

# If SuperGlue produces fewer than this many raw matches, the pipeline tries
# the SIFT fallback. This is intended to recover correspondences in cases
# where SuperGlue struggles, particularly on less-distorted imagery.
FALLBACK_MIN_RAW_MATCHES = 100

# When both SuperGlue and SIFT produce valid geometric fits, prefer the
# solution with the higher inlier ratio. If those are comparable, use the
# lower RMSE as the secondary criterion.
FALLBACK_PREFER = "higher_inlier_ratio_then_lower_rmse"


# ---------------------------------------------------------------------------
# Uniform-distribution reporting
#
# The problem statement requires correspondences to be spatially distributed
# across the image. Tiling is the mechanism used to encourage coverage;
# these settings control the measurement and selection of that coverage.
# ---------------------------------------------------------------------------

# Divide the reference image into an 8 x 8 grid when evaluating how evenly
# the accepted correspondences are distributed.
UNIFORM_GRID = (8, 8)

# RMSE below this threshold is treated as sub-pixel accuracy for the
# pipeline's reported metric.
SUBPIXEL_PASS_PX = 1.0


# ---------------------------------------------------------------------------
# Coarse-to-fine registration
#
# These settings control the initial localization and subsequent refinement.
# The approach combines image metadata, coarse image matching, and finer
# feature matching so that expensive matching is performed only after the
# approximate overlap has been established.
# ---------------------------------------------------------------------------

# Maximum long-side dimension used for coarse preview images.
# Keeping the preview small makes the initial localization practical even
# when the original OHRC/TMC-2 or NAC image is extremely large.
MAX_COARSE_DIM = 1800

# Pyramid levels tested around the estimated ground-sampling-distance ratio.
# The negative/positive octave offsets allow the matcher to tolerate some
# error in the initial scale estimate.
COARSE_PYRAMID_LEVELS = (-1, 0, 1, 2)

# If a coarse image has an extremely small short side, it can be uniformly
# upscaled to make feature detection more stable.
COARSE_MIN_SHORT_SIDE = 96

# Prevent coarse matching from aggressively enlarging small images.
COARSE_UPSCALE_CAP = 2.5

# Minimum number of geometrically consistent matches required for a coarse
# localization to be considered usable.
#
# This is intentionally lower than the final registration requirement:
# coarse localization only needs enough consensus to identify the correct
# region of interest.
COARSE_MIN_INLIERS = 8

# Maximum number of selected correspondences allowed inside each spatial
# grid cell. This prevents one highly textured region from dominating the
# final correspondence set.
UNIFORM_MAX_PER_CELL = 12

# Enable sub-pixel corner refinement after the initial geometric fit.
SUBPIXEL_REFINE = True

# Window size used by cv2.cornerSubPix during local refinement.
SUBPIXEL_WIN = (5, 5)

# Maximum movement allowed during sub-pixel refinement.
#
# A large displacement usually indicates that the optimizer has moved away
# from the original feature, so such a refinement is rejected.
SUBPIXEL_MAX_SHIFT = 1.5

# If the first SIFT attempt produces too few matches, retry with a slightly
# more permissive Lowe ratio. This is a fallback rather than the primary
# matching threshold.
FINE_SIFT_RELAX_RATIO = 0.80

# Diagnostic threshold for normalized cross-correlation after coarse
# alignment. A low value does not automatically reject the result; it simply
# indicates that the coarse warp may be unreliable.
COARSE_NCC_WARN = 0.15


# ---------------------------------------------------------------------------
# RANSAC / homography (Stage 7)
# ---------------------------------------------------------------------------

# Maximum reprojection error, in pixels, for a correspondence to be treated
# as an inlier during robust geometric fitting.
RANSAC_REPROJ_THRESH_PX = 3.0

# Minimum number of inliers required before accepting a final geometric fit.
MIN_INLIERS = 15

# Minimum fraction of all candidate correspondences that must agree with
# the estimated transformation.
MIN_INLIER_RATIO = 0.05

# Number of spatial segments used by the optional piecewise registration
# stage when a single global homography is not sufficient.
PIECEWISE_N_SEGMENTS = 8

# Minimum number of points required in a segment before attempting to fit
# its local transformation.
PIECEWISE_MIN_PTS_PER_SEGMENT = 8


# ---------------------------------------------------------------------------
# Warping (Stages 8-9)
# ---------------------------------------------------------------------------

# Maximum destination dimension used by tiled_warp().
#
# Large lunar images can easily exceed available RAM if warped in one
# operation, so the final transformation is processed in bounded tiles.
WARP_MAX_DIM = 2000

# Number of pixels by which neighboring warp tiles overlap. The overlap
# helps avoid visible seams and boundary artifacts between independently
# processed tiles.
WARP_OVERLAP = 100