"""
This module is the single place where image metadata gets interpreted.

OHRC and TMC-2 labels come as PDS4 XML files, while NAC/WAC metadata is
stored in a flat JSON file. Instead of making every other module understand
those different formats, we convert them all into the same Python dictionary.

That gives the rest of the pipeline one consistent interface for things like:

    - image dimensions and data type
    - geographic footprint
    - Sun/illumination geometry
    - spacecraft/viewpoint information
    - pixel resolution and other scale information

The three main challenges from the problem statement are represented here:

    Illumination variation
        sun azimuth, sun elevation, incidence angle

    Viewpoint variation
        roll, pitch, yaw, spacecraft altitude

    Scale variation
        pixel resolution

A few NAC geometry fields are normally unavailable directly in the sidecar
metadata. In those cases we deliberately return None rather than inventing
values. More complete spacecraft geometry would normally come from SPICE
or another sensor model.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import LunarSync.config as config


# PDS4 uses names such as "UnsignedLSB2" instead of NumPy's dtype notation.
# Keeping the conversion table here means the rest of the pipeline can work
# with normal NumPy dtype strings.
PDS4_DTYPE_MAP = {
    "UnsignedByte": "u1",
    "SignedByte": "i1",
    "UnsignedLSB2": "<u2",
    "UnsignedMSB2": ">u2",
    "UnsignedLSB4": "<u4",
    "UnsignedMSB4": ">u4",
}


# ---------------------------------------------------------------------------
# Small XML helpers
# ---------------------------------------------------------------------------

def _strip_ns(tag):
    """
    Remove the XML namespace from a tag.

    PDS4 XML files commonly use namespaces, so a tag may look like:

        {some_namespace}Axis_Array

    The rest of this module only cares about the actual tag name, so we strip
    the namespace before comparing it.
    """
    return tag.split("}")[-1]


def _all_tags(root, tag):
    """
    Find every XML element with the requested tag name.

    Using _strip_ns() here keeps the search independent of the particular
    namespace prefix used by a PDS4 label.
    """
    return [el for el in root.iter() if _strip_ns(el.tag) == tag]


def _text(el):
    """
    Safely return an element's text without surrounding whitespace.

    Missing XML elements are common enough that returning None is more useful
    than forcing every caller to check for the element first.
    """
    return el.text.strip() if (el is not None and el.text) else None


def _first_tag_text(root, tag):
    """
    Return the text from the first matching XML element.

    Most metadata fields occur only once, so for those fields we only need the
    first match rather than walking through the whole list manually.
    """
    els = _all_tags(root, tag)
    return _text(els[0]) if els else None


def _child_text(el, tag):
    """
    Find a direct child with the requested tag and return its text.

    This is intentionally limited to direct children because some PDS4
    structures contain repeated nested elements with the same names.
    """
    for c in el:
        if _strip_ns(c.tag) == tag:
            return _text(c)
    return None


# ---------------------------------------------------------------------------
# PDS4 labels -- OHRC / TMC-2 / IIRS
# ---------------------------------------------------------------------------

def parse_pds4_label(xml_path):
    """
    Parse an OHRC, TMC-2, or IIRS PDS4 XML label.

    All three products use the PDS4 structure, so they can be converted into
    the same internal metadata format here.

    The important design choice is that the rest of the pipeline never has
    to know whether a value originally came from XML or JSON.
    """

    root = ET.parse(xml_path).getroot()

    def corners_from(block_tag):
        """
        Extract the four geographic corners from one PDS4 coordinate block.

        Different products/labels may expose the footprint through slightly
        different coordinate blocks, which is why the caller tries more than
        one possible block.
        """

        blocks = _all_tags(root, block_tag)

        if not blocks:
            return None

        b = blocks[0]

        # Convert the children into a simple dictionary so that the required
        # corner fields can be checked without depending on their XML order.
        d = {
            _strip_ns(c.tag): float(c.text)
            for c in b
            if c.text
        }

        needed = [
            "upper_left_latitude",
            "upper_left_longitude",
            "upper_right_latitude",
            "upper_right_longitude",
            "lower_right_latitude",
            "lower_right_longitude",
            "lower_left_latitude",
            "lower_left_longitude",
        ]

        # A partial footprint is not very useful for overlap estimation.
        # Return None instead of silently constructing an incomplete footprint.
        return (
            {k: d[k] for k in needed}
            if all(k in d for k in needed)
            else None
        )

    # Prefer refined coordinates when they are available. Otherwise fall back
    # to the system-level coordinates supplied by the product.
    corners = (
        corners_from("Refined_Corner_Coordinates")
        or corners_from("System_Level_Coordinates")
    )

    # PDS4 stores image dimensions as Axis_Array entries. We turn them into
    # the familiar (Line, Sample) shape used by NumPy and the rest of the
    # registration pipeline.
    shape = {}

    for ax in _all_tags(root, "Axis_Array"):
        name = _child_text(ax, "axis_name")
        elements = _child_text(ax, "elements")

        if name and elements:
            shape[name] = int(elements)

    # Convert the PDS4 data type into something NumPy understands.
    # Unknown types fall back to uint8 rather than breaking metadata loading.
    dtype_str = _first_tag_text(root, "data_type")
    dtype = PDS4_DTYPE_MAP.get(dtype_str, "u1")

    # Some binary products have an explicit byte offset before the image data.
    # If the label doesn't provide one, zero is the natural default.
    offset_els = _all_tags(root, "offset")
    offset = int(float(offset_els[0].text)) if offset_els else 0

    def f(tag):
        """
        Read a numeric XML field as float.

        Metadata is not guaranteed to contain every geometry value, so a
        missing field becomes None instead of an artificial zero.
        """
        v = _first_tag_text(root, tag)
        return float(v) if v is not None else None

    return {
        "img_file": _first_tag_text(root, "file_name"),

        # NumPy uses (rows, columns), while PDS4 describes these as
        # Line and Sample.
        "shape": (
            shape.get("Line"),
            shape.get("Sample"),
        ),

        "dtype": dtype,
        "offset": offset,

        # Geographic footprint used later for coarse overlap estimation.
        "corners": corners,

        "projection": _first_tag_text(root, "projection"),
        "area": _first_tag_text(root, "area"),

        # ---------------------------------------------------------------
        # Illumination variation
        # ---------------------------------------------------------------
        # These values tell us how the Sun illuminated the surface when
        # the image was captured. Different illumination can make the same
        # terrain look very different even when the geometry is similar.
        "sun_azimuth_deg": f("sun_azimuth"),
        "sun_elevation_deg": f("sun_elevation"),
        "incidence_angle_deg": f("solar_incidence"),

        # ---------------------------------------------------------------
        # Viewpoint variation
        # ---------------------------------------------------------------
        # These describe the camera/spacecraft geometry when available.
        "roll_deg": f("roll"),
        "pitch_deg": f("pitch"),
        "yaw_deg": f("yaw"),
        "spacecraft_altitude_km": f("spacecraft_altitude"),
        "orbit_limb_direction": _first_tag_text(
            root,
            "orbit_limb_direction",
        ),

        # ---------------------------------------------------------------
        # Scale variation
        # ---------------------------------------------------------------
        "pixel_resolution_m": f("pixel_resolution"),
        "focal_length_mm": f("focal_length"),
        "detector_pixel_width_um": f("detector_pixel_width"),

        # Useful for understanding whether the geographic coordinates came
        # from refined/reference data or only the system-level metadata.
        "reference_data_used": _first_tag_text(
            root,
            "reference_data_used",
        ),
    }


# ---------------------------------------------------------------------------
# NAC / WAC -- flat meta.json
# ---------------------------------------------------------------------------

def parse_nac_meta(json_path):
    """
    Parse a NAC/WAC meta.json sidecar into the same dictionary format used
    by the PDS4 products.

    This keeps NAC and WAC compatible with the same downstream pipeline even
    though their metadata source is completely different from OHRC/TMC-2.
    """

    d = json.loads(
        Path(json_path).read_text()
    )

    # NAC/WAC products can be stored at different bit depths. The sidecar
    # tells us how to interpret the image samples.
    sample_bits = d.get("sample_bits", 8)
    dtype = "u1" if sample_bits == 8 else "<u2"

    def corners_from_dict():
        """
        Return the geographic footprint if it exists.

        Keep the original dictionary structure because the overlap module
        already expects these named corner fields.
        """
        c = d.get("corners")
        return c if c else None

    return {
        "img_file": d.get("product_id"),

        "shape": (
            d.get("shape_lines"),
            d.get("shape_samples"),
        ),

        "dtype": dtype,
        "offset": 0,
        "corners": corners_from_dict(),

        "projection": d.get("projection"),
        "area": None,

        # ---------------------------------------------------------------
        # Illumination variation
        # ---------------------------------------------------------------
        "sun_azimuth_deg": d.get("sub_solar_azimuth_deg"),

        # The NAC sidecar normally doesn't provide this directly.
        # We leave it missing rather than pretending it was measured.
        "sun_elevation_deg": None,

        "incidence_angle_deg": d.get("incidence_angle_deg"),

        # ---------------------------------------------------------------
        # Viewpoint variation
        # ---------------------------------------------------------------
        # These values are generally obtained from spacecraft geometry/SPICE
        # rather than this simple sidecar file.
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "spacecraft_altitude_km": None,
        "orbit_limb_direction": None,

        # ---------------------------------------------------------------
        # Scale variation
        # ---------------------------------------------------------------
        "pixel_resolution_m": d.get("resolution_m"),
        "focal_length_mm": None,
        "detector_pixel_width_um": None,

        # Not supplied by this metadata format.
        "reference_data_used": None,
    }


# ---------------------------------------------------------------------------
# Metadata dispatcher
# ---------------------------------------------------------------------------

def load_metadata(sensor, meta_path):
    """
    Load metadata using the parser appropriate for the sensor.

    The rest of the project calls this function instead of having to know
    whether a particular product uses PDS4 XML or JSON metadata.
    """

    # Chandrayaan-2 optical products use PDS4 labels here.
    if sensor in ("ohrc", "tmc", "iirs"):
        return parse_pds4_label(meta_path)

    # LRO camera metadata is handled through the flat sidecar format.
    if sensor in ("nac", "wac"):
        return parse_nac_meta(meta_path)

    raise ValueError(
        f"Unknown sensor '{sensor}' -- expected one of: "
        "ohrc, tmc, iirs, nac, wac"
    )


# ---------------------------------------------------------------------------
# Stage 0 risk check
# ---------------------------------------------------------------------------

def check_illumination_risk(meta):
    """
    Give a simple warning when the Sun elevation is particularly low.

    Low Sun elevation generally means longer shadows and stronger appearance
    changes across the terrain. This does NOT reject the image or stop the
    pipeline; it simply warns us that feature matching may be harder.
    """

    elev = meta.get("sun_elevation_deg")

    # Some products don't provide Sun elevation directly. In that case we
    # cannot make the risk assessment, so report that explicitly.
    if elev is None:
        return (
            False,
            "sun elevation not available for this source -- "
            "cannot assess risk",
        )

    if elev < config.LOW_SUN_ELEVATION_DEG:
        return (
            True,
            f"low sun elevation ({elev:.1f} deg) -- expect strong shadows "
            f"and reduced feature-matching reliability",
        )

    return (
        False,
        f"sun elevation {elev:.1f} deg -- "
        f"no illumination risk flagged",
    )


def print_geometry_summary(meta, label):
    """
    Print a compact summary of the metadata relevant to registration.

    This is mainly a debugging/sanity-check helper. Before running a large
    registration job, it is useful to see the geometry values rather than
    discovering a bad or missing metadata field several stages later.
    """

    print(
        f"=== {label} "
        f"({meta.get('name', meta.get('img_file', '?'))}) ==="
    )

    # ---------------------------------------------------------------
    # Illumination
    # ---------------------------------------------------------------
    print("Illumination variation:")
    print(
        f"  sun azimuth      : "
        f"{meta.get('sun_azimuth_deg')} deg"
    )
    print(
        f"  sun elevation    : "
        f"{meta.get('sun_elevation_deg')} deg"
    )
    print(
        f"  solar incidence  : "
        f"{meta.get('incidence_angle_deg')} deg"
    )

    # ---------------------------------------------------------------
    # Viewpoint
    # ---------------------------------------------------------------
    print("Viewpoint variation:")

    roll = meta.get("roll_deg")
    pitch = meta.get("pitch_deg")
    yaw = meta.get("yaw_deg")

    # If all three are missing, don't print misleading zeros. Explicitly
    # showing "not present" makes missing spacecraft geometry obvious.
    if roll is None and pitch is None and yaw is None:
        print(
            "  roll/pitch/yaw   : not present "
            "(typical for NAC -- SPICE-derived, not in the label)"
        )
    else:
        print(f"  roll             : {roll} deg")
        print(f"  pitch            : {pitch} deg")
        print(f"  yaw              : {yaw} deg")

    print(
        f"  altitude         : "
        f"{meta.get('spacecraft_altitude_km')} km"
    )

    # ---------------------------------------------------------------
    # Scale
    # ---------------------------------------------------------------
    print("Scale variation:")
    print(
        f"  pixel resolution : "
        f"{meta.get('pixel_resolution_m')} m/pixel"
    )

    # Finish with the illumination warning so it is visible immediately
    # before the registration run begins.
    is_risky, message = check_illumination_risk(meta)

    flag = "WARNING" if is_risky else "ok"

    print(f"[{flag}] {message}\n")