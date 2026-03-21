"""
Unit tests for STEP/3MF file conversion and validation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from sensing_voxel.core.models import FileValidationStatus, ModelFileType
from sensing_voxel.slicer.converter import (
    ConversionResult,
    FileFormatValidator,
    Step3MFConverter,
    is_conversion_supported,
)


class TestModelFileType:
    """Test ModelFileType enum."""

    def test_from_extension_stl(self):
        assert ModelFileType.from_extension("stl") == ModelFileType.STL
        assert ModelFileType.from_extension(".stl") == ModelFileType.STL

    def test_from_extension_step(self):
        assert ModelFileType.from_extension("step") == ModelFileType.STEP
        assert ModelFileType.from_extension(".step") == ModelFileType.STEP

    def test_from_extension_stp(self):
        assert ModelFileType.from_extension("stp") == ModelFileType.STP
        assert ModelFileType.from_extension(".stp") == ModelFileType.STP

    def test_from_extension_3mf(self):
        assert ModelFileType.from_extension("3mf") == ModelFileType.THREE_MF
        assert ModelFileType.from_extension(".3mf") == ModelFileType.THREE_MF

    def test_from_extension_gcode(self):
        assert ModelFileType.from_extension("gcode") == ModelFileType.GCODE
        assert ModelFileType.from_extension("g") == ModelFileType.GCODE
        assert ModelFileType.from_extension("gco") == ModelFileType.GCODE

    def test_from_extension_unknown(self):
        assert ModelFileType.from_extension("obj") is None
        assert ModelFileType.from_extension("ply") is None

    def test_is_slicable(self):
        assert ModelFileType.STL.is_slicable is True
        assert ModelFileType.STEP.is_slicable is True
        assert ModelFileType.STP.is_slicable is True
        assert ModelFileType.THREE_MF.is_slicable is True
        assert ModelFileType.GCODE.is_slicable is False


class TestFileFormatValidator:
    """Test FileFormatValidator."""

    def test_detect_format_step(self, tmp_path):
        # Create a minimal STEP file header
        step_file = tmp_path / "test.step"
        step_file.write_bytes(b"ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n")

        result = FileFormatValidator.detect_format(step_file)
        assert result == "step"

    def test_detect_format_3mf(self, tmp_path):
        import zipfile

        # Create a minimal 3MF file
        three_mf_file = tmp_path / "test.3mf"
        with zipfile.ZipFile(three_mf_file, "w") as zf:
            zf.writestr("3D/3dmodel.model", "<model>")
            zf.writestr("[Content_Types].xml", "<Types>")

        result = FileFormatValidator.detect_format(three_mf_file)
        assert result == "3mf"

    def test_detect_format_invalid(self, tmp_path):
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not a valid file")

        result = FileFormatValidator.detect_format(invalid_file)
        assert result is None

    def test_is_valid_3mf_valid(self, tmp_path):
        import zipfile

        three_mf_file = tmp_path / "test.3mf"
        with zipfile.ZipFile(three_mf_file, "w") as zf:
            zf.writestr("3D/3dmodel.model", "<model>")
            zf.writestr("[Content_Types].xml", "<Types>")

        assert FileFormatValidator._is_valid_3mf(three_mf_file) is True

    def test_is_valid_3mf_missing_model(self, tmp_path):
        import zipfile

        zip_file = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("other.txt", "content")

        assert FileFormatValidator._is_valid_3mf(zip_file) is False

    def test_validate_step_valid(self, tmp_path):
        step_file = tmp_path / "test.step"
        step_file.write_bytes(b"ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n")

        is_valid, warnings = FileFormatValidator.validate_step(step_file)
        assert is_valid is True
        assert len(warnings) == 0

    def test_validate_step_invalid_header(self, tmp_path):
        step_file = tmp_path / "test.step"
        step_file.write_text("not a step file")

        is_valid, warnings = FileFormatValidator.validate_step(step_file)
        assert is_valid is False
        assert "Missing ISO-10303-21 header" in warnings


class TestStep3MFConverter:
    """Test Step3MFConverter."""

    def test_init(self, tmp_path):
        converter = Step3MFConverter(tmp_path)
        assert converter.output_dir == tmp_path
        assert tmp_path.exists()

    def test_check_trimesh_not_available(self):
        with patch(
            "sensing_voxel.slicer.converter.Step3MFConverter._check_trimesh", return_value=False
        ):
            converter = Step3MFConverter()
            assert converter._trimesh_available is False

    @pytest.mark.asyncio
    async def test_convert_to_stl_file_not_found(self, tmp_path):
        converter = Step3MFConverter(tmp_path)
        result = await converter.convert_to_stl(tmp_path / "nonexistent.step")

        assert result.success is False
        assert "File not found" in result.errors[0]

    @pytest.mark.asyncio
    async def test_convert_to_stl_unknown_format(self, tmp_path):
        converter = Step3MFConverter(tmp_path)

        # Create a file with unsupported extension
        unknown_file = tmp_path / "test.unknown"
        unknown_file.write_text("content")

        result = await converter.convert_to_stl(unknown_file)

        assert result.success is False
        assert "Unknown or unsupported file format" in result.errors[0]

    @pytest.mark.asyncio
    async def test_convert_step_no_trimesh(self, tmp_path):
        converter = Step3MFConverter(tmp_path)
        converter._trimesh_available = False

        step_file = tmp_path / "test.step"
        step_file.write_bytes(b"ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n")

        result = await converter.convert_to_stl(step_file)

        assert result.success is False
        assert "trimesh required" in result.errors[0]

    @pytest.mark.asyncio
    async def test_convert_3mf_no_trimesh(self, tmp_path):
        import zipfile

        converter = Step3MFConverter(tmp_path)
        converter._trimesh_available = False

        three_mf_file = tmp_path / "test.3mf"
        with zipfile.ZipFile(three_mf_file, "w") as zf:
            zf.writestr("3D/3dmodel.model", "<model>")
            zf.writestr("[Content_Types].xml", "<Types>")

        result = await converter.convert_to_stl(three_mf_file)

        assert result.success is False
        assert "trimesh required" in result.errors[0]

    def test_extract_3mf_metadata(self, tmp_path):
        import zipfile

        converter = Step3MFConverter(tmp_path)

        three_mf_file = tmp_path / "test.3mf"
        with zipfile.ZipFile(three_mf_file, "w") as zf:
            # Proper 3MF XML format
            model_xml = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0"/>
        </vertices>
        <triangles>
          <triangle v1="0" v2="1" v3="2"/>
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>"""
            zf.writestr("3D/3dmodel.model", model_xml)
            zf.writestr("[Content_Types].xml", "<Types>")

        metadata = converter.extract_3mf_metadata(three_mf_file)
        assert "object_count" in metadata


class TestIsConversionSupported:
    """Test is_conversion_supported function."""

    def test_step_supported(self, tmp_path):
        assert is_conversion_supported(tmp_path / "test.step") is True

    def test_stp_supported(self, tmp_path):
        assert is_conversion_supported(tmp_path / "test.stp") is True

    def test_3mf_supported(self, tmp_path):
        assert is_conversion_supported(tmp_path / "test.3mf") is True

    def test_stl_not_supported(self, tmp_path):
        assert is_conversion_supported(tmp_path / "test.stl") is False

    def test_gcode_not_supported(self, tmp_path):
        assert is_conversion_supported(tmp_path / "test.gcode") is False


class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_post_init_defaults(self):
        result = ConversionResult(success=True)
        assert result.warnings == []
        assert result.errors == []
        assert result.metadata == {}

    def test_post_init_with_values(self):
        result = ConversionResult(
            success=True,
            stl_path=Path("/test.stl"),
            warnings=["warning1"],
            errors=["error1"],
            metadata={"key": "value"},
        )
        assert result.warnings == ["warning1"]
        assert result.errors == ["error1"]
        assert result.metadata == {"key": "value"}
