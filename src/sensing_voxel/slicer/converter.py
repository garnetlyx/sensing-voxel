"""
STEP/3MF file validation and conversion utilities.

Handles conversion of CAD formats (STEP, 3MF) to STL for slicing pipeline.
Uses trimesh for mesh operations and python-opencascade for STEP parsing.
"""

from __future__ import annotations

import logging
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of file conversion to STL."""

    success: bool
    stl_path: Path | None = None
    original_type: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FileFormatValidator:
    """Validate STEP and 3MF file formats."""

    # Magic bytes for file format detection
    ZIP_MAGIC = b"PK\x03\x04"  # ZIP/3MF files start with this
    STEP_MAGIC = b"ISO-10303-21"  # STEP files start with this header

    @classmethod
    def detect_format(cls, file_path: Path) -> str | None:
        """Detect file format from magic bytes and content."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(256)

            # Check for ZIP (3MF)
            if header.startswith(cls.ZIP_MAGIC):
                if cls._is_valid_3mf(file_path):
                    return "3mf"
                return None

            # Check for STEP
            if header.startswith(cls.STEP_MAGIC) or b"ISO-10303-21" in header[:512]:
                return "step"

            return None
        except Exception as e:
            logger.error(f"Error detecting format for {file_path}: {e}")
            return None

    @classmethod
    def _is_valid_3mf(cls, file_path: Path) -> bool:
        """Validate 3MF file structure."""
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                # Check for required 3MF structure
                namelist = zf.namelist()

                # Must have 3D model file
                if "3D/3dmodel.model" not in namelist:
                    return False

                # Check for content types
                if "[Content_Types].xml" not in namelist:
                    return False

                return True
        except zipfile.BadZipFile:
            return False
        except Exception as e:
            logger.error(f"Error validating 3MF {file_path}: {e}")
            return False

    @classmethod
    def validate_step(cls, file_path: Path) -> tuple[bool, list[str]]:
        """Validate STEP file format and structure."""
        warnings = []

        try:
            with open(file_path, "rb") as f:
                content = f.read(4096)

            # Check STEP header
            if b"ISO-10303-21" not in content:
                return False, ["Missing ISO-10303-21 header"]

            # Check for required sections
            with open(file_path, "r", errors="replace") as f:
                content_text = f.read(8192)

            if "HEADER" not in content_text:
                return False, ["Missing HEADER section"]

            if "ENDSEC" not in content_text:
                return False, ["Missing ENDSEC terminator"]

            return True, warnings

        except Exception as e:
            return False, [f"Validation error: {e}"]


class Step3MFConverter:
    """Convert STEP and 3MF files to STL format."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("data/converted")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._trimesh_available = self._check_trimesh()

    def _check_trimesh(self) -> bool:
        """Check if trimesh is available."""
        try:
            import trimesh

            return True
        except ImportError:
            logger.warning("trimesh not installed. STEP/3MF conversion limited.")
            return False

    async def convert_to_stl(
        self,
        input_path: Path,
        output_name: str | None = None,
        target_ext: str = ".stl",
    ) -> ConversionResult:
        """
        Convert STEP or 3MF file to STL.

        Args:
            input_path: Path to input file
            output_name: Optional output filename (defaults to input name)
            target_ext: Target extension (default .stl)

        Returns:
            ConversionResult with success status and output path
        """
        if not input_path.exists():
            return ConversionResult(success=False, errors=[f"File not found: {input_path}"])

        # Detect format
        detected_format = FileFormatValidator.detect_format(input_path)
        if not detected_format:
            return ConversionResult(success=False, errors=["Unknown or unsupported file format"])

        # Generate output path
        stem = output_name or input_path.stem
        stl_path = self.output_dir / f"{stem}{target_ext}"

        try:
            if detected_format == "3mf":
                return await self._convert_3mf(input_path, stl_path)
            elif detected_format == "step":
                return await self._convert_step(input_path, stl_path)
            else:
                return ConversionResult(
                    success=False, errors=[f"Unsupported format: {detected_format}"]
                )
        except Exception as e:
            logger.error(f"Conversion error: {e}", exc_info=True)
            return ConversionResult(success=False, errors=[str(e)])

    async def _convert_3mf(self, input_path: Path, output_path: Path) -> ConversionResult:
        """Convert 3MF to STL using trimesh."""
        result = ConversionResult(success=False, original_type="3mf")

        if not self._trimesh_available:
            result.errors.append("trimesh required for 3MF conversion")
            return result

        try:
            import trimesh

            # Load 3MF file
            mesh = trimesh.load(str(input_path), file_type="3mf")

            if mesh is None:
                result.errors.append("Failed to load 3MF mesh")
                return result

            # Handle scene (multiple objects)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Export to STL
            mesh.export(str(output_path))

            result.success = True
            result.stl_path = output_path
            result.warnings.append("Converted from 3MF to STL")

            # Extract metadata
            result.metadata["vertex_count"] = len(mesh.vertices)
            result.metadata["face_count"] = len(mesh.faces)
            result.metadata["volume_mm3"] = float(mesh.volume) if mesh.is_watertight else 0.0

        except Exception as e:
            result.errors.append(f"3MF conversion failed: {e}")
            logger.error(f"3MF conversion error: {e}", exc_info=True)

        return result

    async def _convert_step(self, input_path: Path, output_path: Path) -> ConversionResult:
        """Convert STEP to STL using trimesh."""
        result = ConversionResult(success=False, original_type="step")

        if not self._trimesh_available:
            result.errors.append("trimesh required for STEP conversion")
            return result

        try:
            import trimesh

            # Load STEP file using trimesh's STEP loader
            mesh = trimesh.load(str(input_path), file_type="step")

            if mesh is None:
                result.errors.append("Failed to load STEP mesh")
                return result

            # Handle scene
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Export to STL
            mesh.export(str(output_path))

            result.success = True
            result.stl_path = output_path
            result.warnings.append("Converted from STEP to STL")

            # Extract metadata
            result.metadata["vertex_count"] = len(mesh.vertices)
            result.metadata["face_count"] = len(mesh.faces)
            result.metadata["volume_mm3"] = float(mesh.volume) if mesh.is_watertight else 0.0

        except Exception as e:
            result.errors.append(f"STEP conversion failed: {e}")
            logger.error(f"STEP conversion error: {e}", exc_info=True)

        return result

    def extract_3mf_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract metadata from 3MF file."""
        metadata = {}

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                # Parse 3D model XML
                if "3D/3dmodel.model" in zf.namelist():
                    with zf.open("3D/3dmodel.model") as f:
                        tree = ET.parse(f)
                        root = tree.getroot()

                        # Extract model info
                        model_elem = root.find(
                            ".{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}model"
                        )
                        if model_elem is not None:
                            metadata["unit"] = model_elem.get("unit", "millimeter")

                        # Count objects
                        objects = root.findall(
                            ".//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}object"
                        )
                        metadata["object_count"] = len(objects)

                # Check for thumbnail
                if "Metadata/thumbnail.png" in zf.namelist():
                    metadata["has_thumbnail"] = True

                # Check for print tickets
                if "3D/PrintTicket.xml" in zf.namelist():
                    metadata["has_print_ticket"] = True

        except Exception as e:
            logger.warning(f"Failed to extract 3MF metadata: {e}")

        return metadata

    def extract_step_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract metadata from STEP file header."""
        metadata = {}

        try:
            with open(file_path, "r", errors="replace") as f:
                content = f.read(4096)

            # Parse HEADER section
            header_match = None
            for line in content.split(";"):
                if "FILE_NAME" in line:
                    # Extract filename and metadata
                    parts = line.split("(")
                    if len(parts) >= 2:
                        metadata["file_name"] = parts[1].split(",")[0].strip("'")
                elif "FILE_DESCRIPTION" in line:
                    desc_parts = line.split("(")
                    if len(desc_parts) >= 2:
                        metadata["description"] = desc_parts[1].split(")")[0].strip("'")
                elif "FILE_SCHEMA" in line:
                    schema_parts = line.split("(")
                    if len(schema_parts) >= 2:
                        metadata["schema"] = (
                            schema_parts[1].split("'")[1] if "'" in schema_parts[1] else ""
                        )

        except Exception as e:
            logger.warning(f"Failed to extract STEP metadata: {e}")

        return metadata


def is_conversion_supported(file_path: Path) -> bool:
    """Check if a file can be converted (STEP or 3MF)."""
    ext = file_path.suffix.lower()
    return ext in {".step", ".stp", ".3mf"}
