"""
Stage 10 of the pipeline: turning the fit result and warped image into the
numbers the problem statement actually asks for.

This module collects the main registration-quality metrics:

- RMSE and its x/y components for geometric reprojection error
- Inlier count and inlier ratio from the RANSAC fit
- Median reprojection error as a robust companion to RMSE
- Uniformity coverage of the final correspondences
- PSNR and SSIM for image-level similarity
- Per-stage and total execution time

The final result is assembled into a single dictionary so it can be saved
directly as JSON and used for reports, comparisons, or demonstrations.
"""

import time

import numpy as np
import cv2


class StageTimer:
    """
    Small helper for measuring how long individual pipeline stages take.

    The pipeline has several independent processing stages, so keeping their
    timing in one place makes it easier to identify expensive operations
    without adding decorators or context-manager code around every stage.

    Usage:
        timer = StageTimer()
        timer.start("preprocessing")
        ... do work ...
        timer.stop("preprocessing")
        ...
        timer.as_dict()  -> {"preprocessing": 12.3, ...}
    """

    def __init__(self):
        # Store the start time separately from the completed durations.
        # This also allows multiple stages to be timed independently.
        self._starts = {}
        self._durations = {}

    def start(self, stage_name):
        """Record the start time for a pipeline stage."""
        self._starts[stage_name] = time.time()

    def stop(self, stage_name):
        """
        Stop timing a stage and store its elapsed time.

        A missing start is treated as a programming error rather than
        silently returning a meaningless duration.
        """
        if stage_name not in self._starts:
            raise KeyError(
                f"StageTimer.stop('{stage_name}') called "
                f"without a matching start()."
            )

        self._durations[stage_name] = (
            time.time() - self._starts[stage_name]
        )

    def as_dict(self):
        """Return a copy of the recorded stage durations."""
        return dict(self._durations)

    def total_seconds(self):
        """Return the sum of all recorded stage durations."""
        return sum(self._durations.values())


class Metrics:
    """
    RAM-safe PSNR/SSIM calculation between a reference image and its warped
    counterpart.

    MSE is accumulated in row chunks instead of creating one full-resolution
    float64 difference image. This matters for large lunar rasters, where
    temporarily allocating another full-size floating-point image can consume
    a significant amount of memory.

    SSIM is calculated on a bounded-size downsample so that its local filtering
    remains practical even when the input image is very large.
    """

    def __init__(
        self,
        image_a,
        image_b,
        ssim_max_dim=3000,
        chunk_rows=512
    ):
        # Pixel-wise metrics require the two images to correspond spatially,
        # so comparing different shapes would not be meaningful.
        if image_a.shape != image_b.shape:
            raise ValueError(
                f"Metric images must have the same shape: "
                f"{image_a.shape} vs {image_b.shape}"
            )

        self.image_a = image_a
        self.image_b = image_b

        # Keep the limits configurable because the appropriate trade-off
        # between speed, memory usage and metric resolution can depend on
        # the image size.
        self.ssim_max_dim = int(ssim_max_dim)
        self.chunk_rows = int(chunk_rows)

    def mse(self):
        """
        Calculate mean squared error without allocating a full-size
        floating-point difference array.
        """

        h = self.image_a.shape[0]

        total = 0.0
        count = 0

        # Process the image a few hundred rows at a time. The temporary
        # arrays created inside this loop are therefore bounded in size.
        for y0 in range(0, h, self.chunk_rows):
            y1 = min(
                y0 + self.chunk_rows,
                h
            )

            a = np.asarray(
                self.image_a[y0:y1],
                dtype=np.float32
            )

            b = np.asarray(
                self.image_b[y0:y1],
                dtype=np.float32
            )

            diff = a - b

            # Accumulate using float64 so that the running sum does not lose
            # precision for large images with many pixels.
            total += float(
                np.sum(
                    diff * diff,
                    dtype=np.float64
                )
            )

            count += diff.size

        return total / max(count, 1)

    def psnr(self):
        """
        Calculate PSNR from the mean squared error.

        For identical images, MSE is zero and PSNR is mathematically
        unbounded, so infinity is returned.
        """

        mse_value = self.mse()

        if mse_value <= 0:
            return float("inf")

        # The implementation assumes an 8-bit intensity range of 0-255.
        return float(
            20.0
            * np.log10(
                255.0 / np.sqrt(mse_value)
            )
        )

    def ssim(self):
        """
        Calculate a simplified single-channel SSIM score.

        Large images are downsampled before the local statistics are computed.
        This keeps the filtering operation manageable while retaining the
        overall structural similarity signal.
        """

        a = np.asarray(self.image_a)
        b = np.asarray(self.image_b)

        # Keep the largest image dimension at or below ssim_max_dim.
        # INTER_AREA is used because it is designed for image reduction.
        scale = min(
            1.0,
            self.ssim_max_dim / max(a.shape)
        )

        if scale < 1.0:
            new_w = max(
                32,
                int(round(a.shape[1] * scale))
            )

            new_h = max(
                32,
                int(round(a.shape[0] * scale))
            )

            a = cv2.resize(
                a,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            b = cv2.resize(
                b,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        # Floating-point arithmetic is needed for the local mean and
        # variance calculations below.
        a = a.astype(
            np.float32,
            copy=False
        )

        b = b.astype(
            np.float32,
            copy=False
        )

        # Standard SSIM stabilization constants for an 8-bit image range.
        # They prevent divisions from becoming unstable when local variance
        # or intensity values are close to zero.
        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2

        # Build the standard 11x11 Gaussian weighting window.
        kernel = cv2.getGaussianKernel(
            11,
            1.5
        )

        window = kernel @ kernel.T

        # Local mean intensity for both images.
        mu_a = cv2.filter2D(
            a,
            -1,
            window
        )

        mu_b = cv2.filter2D(
            b,
            -1,
            window
        )

        # Local variance estimates.
        sigma_a = (
            cv2.filter2D(a * a, -1, window)
            - mu_a * mu_a
        )

        sigma_b = (
            cv2.filter2D(b * b, -1, window)
            - mu_b * mu_b
        )

        # Local covariance between the two images.
        sigma_ab = (
            cv2.filter2D(a * b, -1, window)
            - mu_a * mu_b
        )

        # SSIM combines local luminance and local structural similarity.
        numerator = (
            (2 * mu_a * mu_b + C1)
            * (2 * sigma_ab + C2)
        )

        denominator = (
            (mu_a * mu_a + mu_b * mu_b + C1)
            * (sigma_a + sigma_b + C2)
        )

        # Protect against numerical division by zero in very uniform regions.
        ssim_map = (
            numerator
            / np.maximum(denominator, 1e-12)
        )

        # Ignore the outer border where the filtering window is incomplete.
        # This avoids letting border effects influence the overall score.
        if (
            ssim_map.shape[0] > 12
            and ssim_map.shape[1] > 12
        ):
            ssim_map = ssim_map[5:-5, 5:-5]

        return float(
            np.mean(
                ssim_map,
                dtype=np.float64
            )
        )


def compute_rmse(
    mkpts_src,
    mkpts_ref,
    H,
    inlier_mask
):
    """
    Calculate geometric RMSE using only RANSAC inlier correspondences.

    Each inlier source point is transformed through H and compared with the
    corresponding reference point.

    Returns:
        {
            "rmse_x": ...,
            "rmse_y": ...,
            "rmse_total": ...
        }

    The total RMSE is the Euclidean pixel error, while rmse_x and rmse_y
    expose the horizontal and vertical components separately.
    """

    # Discard rejected matches before evaluating the transformation.
    points_src = mkpts_src[inlier_mask]
    points_ref = mkpts_ref[inlier_mask]

    if len(points_src) == 0:
        return {
            "rmse_x": None,
            "rmse_y": None,
            "rmse_total": None
        }

    # Map the source inliers into the reference coordinate system.
    projected = cv2.perspectiveTransform(
        points_src.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    # Difference between predicted and observed reference positions.
    diff = projected - points_ref

    # Calculate independent horizontal and vertical RMS errors.
    rmse_x = float(
        np.sqrt(
            np.mean(diff[:, 0] ** 2)
        )
    )

    rmse_y = float(
        np.sqrt(
            np.mean(diff[:, 1] ** 2)
        )
    )

    # For total RMSE, calculate the Euclidean error for each point first,
    # then take the root mean square of those distances.
    rmse_total = float(
        np.sqrt(
            np.mean(
                np.sum(diff ** 2, axis=1)
            )
        )
    )

    return {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_total": rmse_total
    }


def build_report(
    pair_name,
    sensor,
    geometry_summary,
    fit_result,
    rmse,
    psnr,
    ssim,
    timer,
    uniformity=None,
    median_err=None,
    matcher_used="superglue",
    coarse_method="sift_pyramid",
    source_band=None,
    source_pixels="full_img",
    reference_pixels="full_img",
    coarse_ncc=None,
    note=None
):
    """
    Assemble all results from one registration run into a JSON-serializable
    dictionary.

    The report combines geometric metrics, image-similarity metrics,
    correspondence statistics, processing details and timing so that one
    output file contains the complete record of a registration experiment.
    """

    # RMSE normally arrives as the dictionary produced by compute_rmse().
    # Keep this small extraction separate so the report can also tolerate
    # callers that do not provide a complete RMSE dictionary.
    rmse_total = (
        rmse.get("rmse_total")
        if isinstance(rmse, dict)
        else None
    )

    import LunarSync.config as _cfg

    report = {
        "pair_name": pair_name,
        "sensor": sensor,

        # Record which correspondence method produced the final matches.
        # This is important when comparing SuperGlue and SIFT-based runs.
        "matcher_used": matcher_used,

        # Keep track of how the initial coarse localization was obtained.
        "coarse_method": coarse_method,

        # Relevant when the source is a hyperspectral cube such as IIRS.
        "source_band": source_band,

        # Record whether the run used the complete raster or a quick-look
        # preview. This prevents metrics from being confused across runs
        # using different image representations.
        "source_pixels": source_pixels,
        "reference_pixels": reference_pixels,

        # Preserve the metadata/geometry information produced earlier in
        # the pipeline rather than recomputing it here.
        "geometry": geometry_summary,

        # Record whether the final transformation was global or piecewise.
        "fit_mode": fit_result["mode"],

        # These are the primary correspondence-quality statistics.
        "n_inliers": fit_result["n_inliers"],
        "inlier_ratio": fit_result["inlier_ratio"],

        # This is the final quality-gate decision from geometry_fit.py.
        "registration_accepted": fit_result["registration_accepted"],

        # Geometric accuracy metrics.
        "rmse_x": rmse["rmse_x"],
        "rmse_y": rmse["rmse_y"],
        "rmse_total": rmse_total,

        # Median error provides a robust companion to RMSE, particularly
        # when a small number of correspondences have unusually large errors.
        "median_reproj_error_px": median_err,

        # This flag is based on the configured pixel threshold. It should be
        # interpreted as a sub-pixel error criterion, not as independent
        # absolute ground-truth accuracy.
        "subpixel_pass": (
            rmse_total is not None
            and rmse_total < _cfg.SUBPIXEL_PASS_PX
        ),

        # Spatial coverage of the final correspondences.
        "uniformity_coverage": uniformity,

        # NCC is useful for checking whether the coarse ROI looks sensible,
        # but it is deliberately stored as a diagnostic rather than used as
        # the registration acceptance criterion.
        "coarse_ncc": coarse_ncc,

        # Image-level similarity metrics.
        "psnr": psnr,
        "ssim": ssim,

        # Preserve both individual stage timings and the overall runtime.
        "timing_seconds": timer.as_dict(),
        "total_time_seconds": timer.total_seconds(),

        # Unless the caller supplies a more specific note, explicitly state
        # what RMSE does and does not prove. A low reprojection error measures
        # consistency with the fitted transformation, not absolute accuracy
        # against an independently surveyed lunar ground truth.
        "note": note or (
            "RMSE is geometric reprojection error. No absolute ground-truth "
            "accuracy is claimed unless an independently surveyed/validated "
            "ground truth is supplied."
        ),
    }

    return report