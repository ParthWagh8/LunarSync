"""
All filesystem-related work lives here.

This module handles things like:

    - finding image + metadata files
    - memory-mapping large .IMG products
    - loading/saving PNG previews
    - reading/writing JSON reports
    - creating output directories
    - releasing large arrays from memory

The important design idea is that this file knows about files and arrays,
but it does NOT know how registration or matching works.

That keeps the rest of the pipeline focused on image processing rather than
constantly dealing with paths, file formats, or memory management.
"""

import gc
import json
from pathlib import Path

import cv2
import numpy as np


def ensure_dirs(root, subdirs):
    """
    Create the requested output directories under `root`.

    exist_ok=True makes this safe to call at the beginning of every run:
    already-existing folders are simply reused.
    """

    root = Path(root)

    for name in subdirs:
        (root / name).mkdir(
            parents=True,
            exist_ok=True,
        )

    return root


def find_product_files(source_dir, sensor):
    """
    Find the image and corresponding metadata file for one product.

    Expected layouts:

        OHRC / TMC-2 / IIRS
            .IMG + .xml

        NAC / WAC
            .IMG + _meta.json

    Only the supplied directory is searched. This is intentional because
    lunar data downloads can contain many nested folders, and silently
    searching everything can easily pick the wrong product.

    Returns:
        (img_path, meta_path)

    Both values are returned as pathlib.Path objects.
    """

    source_dir = Path(source_dir)

    # PDS products are sometimes written with either .IMG or .img, so check
    # both rather than relying on the exact capitalization used by the archive.
    img_path = (
        next(source_dir.glob("*.IMG"), None)
        or next(source_dir.glob("*.img"), None)
    )

    # The metadata format depends on the sensor family.
    if sensor in ("ohrc", "tmc", "iirs"):
        meta_path = next(
            source_dir.glob("*.xml"),
            None,
        )
        expected = ".xml"

    elif sensor in ("nac", "wac"):
        meta_path = next(
            source_dir.glob("*_meta.json"),
            None,
        )
        expected = "_meta.json"

    else:
        raise ValueError(
            f"Unknown sensor '{sensor}' -- expected one of: "
            "ohrc, tmc, iirs, nac, wac"
        )

    # Fail early with a useful message. It is much easier to fix a missing
    # product file here than to get a cryptic NumPy/OpenCV error several
    # stages later.
    if img_path is None or meta_path is None:
        raise FileNotFoundError(
            f"{sensor} @ {source_dir}: expected one .IMG and one "
            f"{expected} file directly in this folder "
            f"(found img={img_path is not None}, "
            f"meta={meta_path is not None})."
        )

    return img_path, meta_path


def find_preview_png(search_root, include=(), exclude=()):
    """
    Recursively find a preview PNG matching simple filename filters.

    This is mainly used for rapid-look workflows where working with a small
    browse image is much faster than reading a multi-gigabyte raw product.

    If nothing matches, return None rather than raising an error. The caller
    can then decide whether the preview is optional or required.
    """

    search_root = Path(search_root)

    out = []

    for p in sorted(search_root.rglob("*.png")):
        name = p.name.lower()

        # Every requested include string must be present, while none of the
        # excluded strings may appear in the filename.
        if (
            all(s in name for s in include)
            and not any(s in name for s in exclude)
        ):
            out.append(p)

    return out[0] if out else None


def load_pixel_array(img_path, shape, dtype, offset=0):
    """
    Memory-map a raw .IMG product as a 2D NumPy array.

    Memory mapping is important for the large lunar products used here:
    the operating system can expose the image as an array without loading
    the entire file into RAM at once.

    Before creating the memmap, we check that the file is large enough for
    the dimensions and dtype claimed by the metadata. This catches truncated
    downloads and incorrect labels early.
    """

    img_path = Path(img_path)
    dtype = np.dtype(dtype)

    # Calculate exactly how many bytes the metadata says the image should use.
    expected_bytes = (
        int(np.prod(shape))
        * dtype.itemsize
    )

    # The offset is metadata/header bytes that should not be interpreted as
    # image pixels.
    available_bytes = (
        img_path.stat().st_size
        - int(offset)
    )

    if available_bytes < expected_bytes:
        raise ValueError(
            f"{img_path.name}: file too small for the metadata's claimed shape. "
            f"Need {expected_bytes:,} bytes after offset {offset}, "
            f"have {available_bytes:,}. "
            f"Check the label's shape/dtype against the actual .IMG."
        )

    return np.memmap(
        img_path,
        dtype=dtype,
        mode="r",
        offset=int(offset),
        shape=tuple(shape),
    )


def load_source_pixels(img_path, shape, dtype, offset=0):
    """
    Load a source product while allowing for hyperspectral data.

    Normal 2D products such as OHRC, TMC-2 and NAC take the fast memory-map
    path.

    If the file is larger than the expected 2D image, we check whether its
    size is consistent with a banded cube. This allows the same loader to
    handle IIRS-style data without requiring the rest of the pipeline to
    understand the cube layout.

    For a hyperspectral cube, the band with the highest variance is selected
    as a representative image for the registration stage.

    Returns:
        (image_array_or_memmap, selected_band_index)

    For a normal 2D image, the band index is None.
    """

    img_path = Path(img_path)
    dtype = np.dtype(dtype)

    h = int(shape[0])
    w = int(shape[1])

    expected_bytes = (
        h
        * w
        * dtype.itemsize
    )

    available_bytes = (
        img_path.stat().st_size
        - int(offset)
    )

    # If the file is even smaller than the claimed 2D product, let the normal
    # loader produce the more useful validation error.
    if available_bytes < expected_bytes:
        return (
            load_pixel_array(
                img_path,
                shape,
                dtype,
                offset,
            ),
            None,
        )

    # Exact size means this is a normal 2D image. Keep it memory-mapped rather
    # than copying the entire product into RAM.
    if available_bytes == expected_bytes:
        return (
            np.memmap(
                img_path,
                dtype=dtype,
                mode="r",
                offset=int(offset),
                shape=(h, w),
            ),
            None,
        )

    # If the file is larger than the expected 2D image, see whether the
    # remaining bytes divide evenly into complete h x w bands.
    pixel_bytes = (
        h
        * w
        * dtype.itemsize
    )

    if (
        pixel_bytes > 0
        and available_bytes % pixel_bytes == 0
    ):
        n_bands = available_bytes // pixel_bytes

        if n_bands > 1:

            # Different products can arrange the band dimension differently.
            # Try both common layouts:
            #
            #     (height, width, bands)
            #     (bands, height, width)
            #
            # and only continue when NumPy can successfully map the file.
            for order in ("last", "first"):
                try:
                    cube = np.memmap(
                        img_path,
                        dtype=dtype,
                        mode="r",
                        offset=int(offset),
                        shape=(
                            (h, w, n_bands)
                            if order == "last"
                            else (n_bands, h, w)
                        ),
                    )
                except ValueError:
                    continue

                arr = np.asarray(cube)

                # Pick the band with the greatest standard deviation.
                # The idea is simply to choose a band containing enough
                # texture for the geometric matching stage.
                stds = [
                    (
                        float(np.std(arr[:, :, b]))
                        if order == "last"
                        else float(np.std(arr[b]))
                    )
                    for b in range(n_bands)
                ]

                band = int(np.argmax(stds))

                print(
                    f"Hyperspectral cube ({n_bands} bands, {order}-axis); "
                    f"band {band} selected by max variance "
                    f"(std={stds[band]:.1f})."
                )

                band_array = (
                    arr[:, :, band]
                    if order == "last"
                    else arr[band]
                )

                # Make the selected band contiguous. This avoids surprises
                # later when OpenCV or other routines expect conventional
                # contiguous image memory.
                return (
                    np.ascontiguousarray(band_array),
                    band,
                )

    # If the file doesn't match either a normal 2D image or a plausible
    # banded cube, the metadata and actual file are inconsistent.
    raise ValueError(
        f"{img_path.name}: size {available_bytes:,} bytes matches neither 2D "
        f"({h}, {w}) nor a banded cube of it -- check the label."
    )


def load_preview_png(path, full_shape):
    """
    Load a browse/preview PNG and determine how it relates to the full image.

    Preview images are often much smaller than the original product, and the
    reduction may not be identical along the two axes.

    Returns:

        (preview, stride_x, stride_y)

    If the preview is noticeably anisotropic, it is first resampled so that
    its pixels represent approximately square ground sampling. This makes
    feature matching more sensible because SIFT/SuperGlue work in an image
    coordinate system where x/y scale should be comparable.
    """

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read preview PNG at {path}"
        )

    full_h = int(full_shape[0])
    full_w = int(full_shape[1])

    # Estimate how many full-resolution pixels are represented by one
    # preview pixel along each axis.
    sy = full_h / img.shape[0]
    sx = full_w / img.shape[1]

    # If the x and y strides differ by more than 5%, the thumbnail is
    # geometrically stretched. Normalize it before using it for matching.
    if abs(sx - sy) / max(sx, sy) > 0.05:

        iso_w = max(
            8,
            int(round(full_w / sy)),
        )

        img = cv2.resize(
            img,
            (iso_w, img.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        print(
            f"Preview PNG was anisotropic "
            f"(strides x={sx:.2f} y={sy:.2f}); "
            f"normalized to {img.shape} square ground pixels."
        )

        # Recalculate the effective strides after normalization.
        sx = full_w / img.shape[1]
        sy = full_h / img.shape[0]

    return (
        img,
        float(sx),
        float(sy),
    )


def decimated_view(array, max_dim=1800):
    """
    Create a lightweight view of a large image by taking every Nth pixel.

    This is intended for coarse processing, previews, and initial localization
    rather than final registration.

    The important part is that the function returns the stride as well, so
    coordinates found on the decimated image can later be related back to
    the original pixel space.

    Unlike image enhancement, this function does not normalize the pixel
    values. It deliberately leaves the original radiometry alone.
    """

    import math

    h, w = array.shape[:2]

    # Small images are allowed to stay relatively detailed, but the preview
    # is still capped so a large browse image doesn't suddenly consume a huge
    # amount of RAM.
    cap = max(
        max_dim,
        min(max(h, w), 5000),
    )

    # Choose an integer stride large enough to keep the longest side within
    # the requested limit.
    stride = max(
        1,
        int(math.ceil(max(h, w) / cap)),
    )

    # NumPy slicing gives us a cheap decimated view for memory-mapped data
    # without resizing the entire source image.
    view = np.asarray(
        array[::stride, ::stride]
    )

    return view, stride


def free_arrays(*arrays):
    """
    Release references to large arrays and request garbage collection.

    This is useful in long-running Colab sessions where several large image
    arrays may otherwise remain alive between pipeline stages.

    Note that deleting the local references does not modify the original
    files or memory-mapped products.
    """

    for arr in arrays:
        del arr

    gc.collect()


def save_png(array_u8, out_path):
    """
    Save an 8-bit image as a PNG.

    The parent directory is created automatically so callers don't need to
    separately prepare every output folder.
    """

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(out_path),
        array_u8,
    )

    return out_path


def load_png_gray(path):
    """
    Read a PNG as a single-channel grayscale image.

    Keeping this helper separate means the rest of the pipeline doesn't need
    to repeat OpenCV's image-loading boilerplate.
    """

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read PNG at {path}"
        )

    return img


def save_json(data, out_path):
    """
    Save a Python dictionary as readable JSON.

    indent=2 makes the registration report easy to inspect manually, which is
    useful during debugging and also makes the final hackathon output easier
    to demonstrate.
    """

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )

    return out_path


def load_json(path):
    """
    Read a JSON file and return the decoded Python object.
    """

    with open(path) as f:
        return json.load(f)