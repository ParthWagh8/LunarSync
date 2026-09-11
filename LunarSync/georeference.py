"""
georeference.py
================
Stage 1 of the pipeline: Georeferencing.

IMPORTANT SCOPE NOTE, stated plainly rather than glossed over:
This module does NOT perform pixel-level map reprojection between OHRC's
Selenographic projection and NAC's Equirectangular Moon projection -- doing
that properly requires a full cartographic library (e.g. GDAL/rasterio with
a real lunar coordinate-reference-system definition) and is out of scope for
"simple, readable code".

What this module actually does, and why it's still correct for this pipeline:
every corner coordinate in both PDS4 labels and NAC's meta.json is expressed
as plain latitude/longitude (planetocentric), regardless of which map
projection was used to originally lay out the image raster. That shared
lat/lon frame is the thing overlap.py actually relies on to find the common
region, and the final pixel-level geometric alignment between the two rasters
is achieved empirically by SuperGlue + homography fitting (geometry_fit.py),
not by analytical reprojection. So this module's real job is just to record
each source's stated projection and confirm both sides have the lat/lon
corners the rest of the pipeline needs -- it's a documentation-and-safety-
check step, not a coordinate transform.
"""


def get_projection(meta):
    """
    Return the projection name recorded in a source's metadata, if available.

    Keeping this as a small helper makes the projection lookup consistent
    throughout the pipeline and also avoids assuming that every metadata
    dictionary contains a projection field.
    """
    return meta.get("projection")


def verify_common_frame(source_meta, reference_meta):
    """
    Check whether both images provide the geographic information required by
    the overlap stage, while also recording the projection used by each one.

    The two datasets do not need to use the same map projection here. What
    matters at this stage is that both provide their image footprints as
    latitude/longitude coordinates so that overlap.py can estimate where the
    two images cover the same lunar surface.

    Returns a dict:
        {
            "source_projection": ...,
            "reference_projection": ...,
            "projections_match": bool,
            "both_have_corners": bool,
            "ok": bool,
            "note": str,
        }

    `projections_match` being False is expected and NOT an error (e.g. OHRC's
    Selenographic vs NAC's Equirectangular Moon) -- it's recorded for
    traceability, not treated as a failure.

    `both_have_corners` is the actual safety check. Without corner
    coordinates, overlap.py cannot estimate the common geographic region.
    """

    # Read the projection names if the metadata contains them. These values
    # are useful for logging and traceability even though we do not reproject
    # the images at this stage.
    source_proj = get_projection(source_meta)
    reference_proj = get_projection(reference_meta)

    # The overlap stage depends on both images having geographic corner
    # coordinates. A missing footprint means we cannot reliably determine
    # which part of the reference image corresponds to the source image.
    both_have_corners = (
        bool(source_meta.get("corners"))
        and bool(reference_meta.get("corners"))
    )

    if both_have_corners:
        # Different projections are acceptable because the corner coordinates
        # provide a common geographic frame for the coarse overlap calculation.
        # The actual pixel-to-pixel alignment is left to the feature matching
        # and geometric fitting stages later in the pipeline.
        note = (
            "Both sources have corner coordinates in a shared lat/lon frame. "
            "Pixel-level alignment across the two map projections will be handled "
            "empirically downstream by SuperGlue + homography fitting, not by "
            "analytical reprojection here."
        )
    else:
        # Without both footprints, the next stage has no reliable way to crop
        # the reference image to the region covered by the source image.
        note = (
            "Missing corner coordinates on at least one side -- overlap.py will not "
            "be able to compute a footprint crop. Check the label parsing in metadata.py."
        )

    return {
        "source_projection": source_proj,
        "reference_projection": reference_proj,

        # Only report whether the projections match when both projection names
        # are actually available. A missing projection should not be treated
        # as a genuine projection mismatch.
        "projections_match": (
            (source_proj == reference_proj)
            if (source_proj and reference_proj)
            else None
        ),

        # This is the important validity check for the rest of the pipeline.
        "both_have_corners": both_have_corners,
        "ok": both_have_corners,

        # Keep a human-readable explanation so the pipeline output can make
        # it clear why the georeferencing check passed or failed.
        "note": note,
    }