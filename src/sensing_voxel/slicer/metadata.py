"""
G-code metadata extraction for first-layer parameters and bounds.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from sensing_voxel.core.models import FirstLayerBounds, FirstLayerParams

logger = logging.getLogger(__name__)

# PrusaSlicer / OrcaSlicer / Cura comment patterns
COMMENT_PATTERNS = {
    "layer_height": re.compile(r";\s*layer_height\s*=\s*([\d.]+)", re.IGNORECASE),
    "first_layer_height": re.compile(
        r";\s*first_layer_height\s*=\s*([\d.]+)", re.IGNORECASE
    ),
    "first_layer_temperature": re.compile(
        r";\s*(?:first_layer_temperature|temperature)\s*=\s*([\d.]+)", re.IGNORECASE
    ),
    "first_layer_bed_temperature": re.compile(
        r";\s*(?:first_layer_bed_temperature|bed_temperature)\s*=\s*([\d.]+)",
        re.IGNORECASE,
    ),
    "first_layer_speed": re.compile(
        r";\s*(?:first_layer_speed|initial_layer_speed)\s*=\s*([\d.]+)", re.IGNORECASE
    ),
    "filament_used_mm": re.compile(
        r";\s*filament used\s*\[mm\]\s*=\s*([\d.]+)", re.IGNORECASE
    ),
    "filament_used_g": re.compile(
        r";\s*(?:filament used\s*\[g\]|total filament weight)\s*=?\s*([\d.]+)",
        re.IGNORECASE,
    ),
    "estimated_time": re.compile(
        r";\s*(?:estimated[_ ]?(?:printing[_ ]?)?time|TIME)\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    ),
    "total_layers": re.compile(
        r";\s*total layer(?:s| number|_count)\s*[:=]\s*(\d+)", re.IGNORECASE
    ),
}

# G1 move pattern for first-layer bounds parsing
G1_MOVE_PATTERN = re.compile(
    r"G1\s+(?:.*?X([\d.]+))?(?:.*?Y([\d.]+))?(?:.*?E[\d.]+)", re.IGNORECASE
)

# Layer change markers
LAYER_CHANGE_PATTERN = re.compile(
    r";\s*(?:LAYER_CHANGE|AFTER_LAYER_CHANGE|;Z:)", re.IGNORECASE
)


class GcodeMetadataExtractor:
    """Extract print metadata from G-code files."""

    @staticmethod
    def extract_metadata(gcode_path: Path) -> dict:
        """Extract metadata from G-code comment headers."""
        try:
            with open(gcode_path, "r", errors="replace") as f:
                header = f.read(10240)
        except OSError as e:
            logger.error(f"Failed to read G-code file: {e}")
            return {}

        metadata = {}
        for key, pattern in COMMENT_PATTERNS.items():
            m = pattern.search(header)
            if m:
                val = m.group(1).strip()
                if key in ("total_layers", "estimated_time"):
                    metadata[key] = int(val)
                else:
                    metadata[key] = float(val)

        return metadata

    @staticmethod
    def extract_first_layer_params(gcode_path: Path) -> FirstLayerParams | None:
        """Extract first-layer parameters from G-code comments."""
        try:
            with open(gcode_path, "r", errors="replace") as f:
                header = f.read(10240)
        except OSError:
            return None

        def find_float(pattern_key: str) -> float | None:
            m = COMMENT_PATTERNS.get(pattern_key, re.compile("$^")).search(header)
            return float(m.group(1)) if m else None

        layer_height = find_float("first_layer_height") or find_float("layer_height")
        nozzle_temp = find_float("first_layer_temperature")
        bed_temp = find_float("first_layer_bed_temperature")
        speed = find_float("first_layer_speed")

        if layer_height is None or nozzle_temp is None or bed_temp is None or speed is None:
            logger.debug("Incomplete first-layer params in G-code")
            return None

        return FirstLayerParams(
            layer_height=layer_height,
            nozzle_temp=nozzle_temp,
            bed_temp=bed_temp,
            speed=speed,
        )

    @staticmethod
    def extract_first_layer_bounds(gcode_path: Path) -> FirstLayerBounds | None:
        """Extract XY bounding box of first-layer moves from G-code."""
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        found_moves = False
        past_first_layer = False

        try:
            with open(gcode_path, "r", errors="replace") as f:
                for line in f:
                    # Stop after first layer change
                    if LAYER_CHANGE_PATTERN.search(line) and found_moves:
                        past_first_layer = True
                        break

                    m = G1_MOVE_PATTERN.match(line)
                    if m:
                        x_str, y_str = m.group(1), m.group(2)
                        if x_str:
                            x = float(x_str)
                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            found_moves = True
                        if y_str:
                            y = float(y_str)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)
                            found_moves = True
        except OSError:
            return None

        if not found_moves or min_x == float("inf"):
            return None

        return FirstLayerBounds(
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
        )

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
