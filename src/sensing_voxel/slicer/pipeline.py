"""
Slicer pipeline for STL to G-code conversion via PrusaSlicer/OrcaSlicer CLI.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sensing_voxel.config import SlicerBackend, SlicerConfig
from sensing_voxel.core.models import FileValidationStatus, SlicingResult, SlicingStatus
from sensing_voxel.slicer.converter import Step3MFConverter
from sensing_voxel.slicer.metadata import GcodeMetadataExtractor

logger = logging.getLogger(__name__)


@dataclass
class SlicerInfo:
    backend: SlicerBackend
    executable_path: str
    version: str | None = None


class SlicerDetector:
    PRUSASLICER_NAMES = ["prusa-slicer", "prusa-slicer-console", "prusa-slicer.exe"]
    ORCASLICER_NAMES = ["orca-slicer", "orcaslicer", "orca-slicer.exe", "OrcaSlicer"]

    @classmethod
    async def detect_slicer(cls, config: SlicerConfig) -> SlicerInfo:
        if config.backend == SlicerBackend.PRUSASLICER:
            return await cls._detect_prusaslicer(config)
        elif config.backend == SlicerBackend.ORCASLICER:
            return await cls._detect_orcaslicer(config)
        else:
            return await cls._auto_detect(config)

    @classmethod
    async def _auto_detect(cls, config: SlicerConfig) -> SlicerInfo:
        try:
            orca_info = await cls._detect_orcaslicer(config)
            return orca_info
        except FileNotFoundError:
            pass

        try:
            prusa_info = await cls._detect_prusaslicer(config)
            return prusa_info
        except FileNotFoundError:
            pass

        raise FileNotFoundError("No slicer found. Install PrusaSlicer or OrcaSlicer.")

    @classmethod
    async def _detect_prusaslicer(cls, config: SlicerConfig) -> SlicerInfo:
        path = config.prusaslicer_path

        if not await cls._executable_exists(path):
            for name in cls.PRUSASLICER_NAMES:
                if shutil.which(name):
                    path = name
                    break

        if not await cls._executable_exists(path):
            raise FileNotFoundError(f"PrusaSlicer not found: {path}")

        version = await cls._get_version(path, "--version")
        return SlicerInfo(backend=SlicerBackend.PRUSASLICER, executable_path=path, version=version)

    @classmethod
    async def _detect_orcaslicer(cls, config: SlicerConfig) -> SlicerInfo:
        path = config.orcaslicer_path

        if not await cls._executable_exists(path):
            for name in cls.ORCASLICER_NAMES:
                if shutil.which(name):
                    path = name
                    break

        if not await cls._executable_exists(path):
            raise FileNotFoundError(f"OrcaSlicer not found: {path}")

        version = await cls._get_version(path, "--version")
        return SlicerInfo(backend=SlicerBackend.ORCASLICER, executable_path=path, version=version)

    @classmethod
    async def _executable_exists(cls, path: str) -> bool:
        if shutil.which(path):
            return True
        return Path(path).exists() and Path(path).is_file()

    @classmethod
    async def _get_version(cls, path: str, version_flag: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                path,
                version_flag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode().strip().split("\n")[0] if stdout else None
        except Exception:
            return None


class ProfileMapper:
    @staticmethod
    def map_profile_path(profile_path: Path | None, target_backend: SlicerBackend) -> Path | None:
        if profile_path is None:
            return None

        if not profile_path.exists():
            return profile_path

        ext = profile_path.suffix.lower()

        if target_backend == SlicerBackend.ORCASLICER:
            if ext == ".json":
                return profile_path
            json_path = profile_path.with_suffix(".json")
            if json_path.exists():
                return json_path
            return profile_path

        elif target_backend == SlicerBackend.PRUSASLICER:
            if ext == ".ini":
                return profile_path
            ini_path = profile_path.with_suffix(".ini")
            if ini_path.exists():
                return ini_path
            return profile_path

        return profile_path

    @staticmethod
    def get_profile_type(profile_path: Path | None) -> str | None:
        if profile_path is None:
            return None
        ext = profile_path.suffix.lower()
        if ext == ".json":
            return "orca"
        elif ext == ".ini":
            return "prusaslicer"
        return "unknown"


class SlicerPipeline:
    def __init__(self, config: SlicerConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = Step3MFConverter(self.output_dir)
        self._slicer_info: SlicerInfo | None = None

    async def _get_slicer(self) -> SlicerInfo:
        if self._slicer_info is None:
            self._slicer_info = await SlicerDetector.detect_slicer(self.config)
        return self._slicer_info

    def _build_slice_command(
        self,
        slicer_info: SlicerInfo,
        stl_path: Path,
        gcode_path: Path,
        profile: Path | None = None,
    ) -> list[str]:
        if slicer_info.backend == SlicerBackend.ORCASLICER:
            cmd = [
                slicer_info.executable_path,
                "slice",
                "--export-gcode",
                str(stl_path),
                "--output",
                str(gcode_path),
            ]
            if profile and profile.exists():
                cmd.extend(["--load-settings", str(profile)])
        else:
            cmd = [
                slicer_info.executable_path,
                "--export-gcode",
                str(stl_path),
                "--output",
                str(gcode_path),
            ]
            if profile and profile.exists():
                cmd.extend(["--load", str(profile)])
        return cmd

    async def process_model(
        self,
        model_path: Path,
        profile_path: Path | None = None,
        output_name: str | None = None,
        timeout_seconds: int = 600,
    ) -> SlicingResult:
        """
        Process a model file (STL, STEP, or 3MF) to G-code.
        Automatically converts STEP/3MF to STL before slicing.

        Args:
            model_path: Path to input model file
            profile_path: Override slicer profile (default from config)
            output_name: Override output filename (default from model name)
            timeout_seconds: Subprocess timeout

        Returns:
            SlicingResult with gcode_path, metadata, and any warnings/errors
        """
        if not model_path.exists():
            return SlicingResult(
                gcode_path=Path(""),
                gcode_hash="",
                errors=[f"Model file not found: {model_path}"],
            )

        ext = model_path.suffix.lower()

        # Handle STEP/3MF conversion
        if ext in {".step", ".stp", ".3mf"}:
            logger.info(f"Converting {model_path.name} from {ext} to STL")

            conversion = await self.converter.convert_to_stl(model_path, output_name=output_name)

            if not conversion.success:
                errors = [f"Failed to convert {ext} file to STL"] + conversion.errors
                return SlicingResult(
                    gcode_path=Path(""),
                    gcode_hash="",
                    errors=errors,
                )

            # Use converted STL path
            stl_path = conversion.stl_path
            if stl_path is None:
                return SlicingResult(
                    gcode_path=Path(""),
                    gcode_hash="",
                    errors=["Conversion succeeded but no STL path returned"],
                )
            conversion_warnings = conversion.warnings
        else:
            # Already STL or other supported format
            stl_path = model_path
            conversion_warnings = []

        # Slice the STL
        result = await self.slice_stl(
            stl_path=stl_path,
            profile_path=profile_path,
            output_name=output_name,
            timeout_seconds=timeout_seconds,
        )

        # Add conversion warnings to result
        if conversion_warnings:
            result.warnings.extend(conversion_warnings)

        return result

    async def slice_stl(
        self,
        stl_path: Path,
        profile_path: Path | None = None,
        output_name: str | None = None,
        timeout_seconds: int = 600,
    ) -> SlicingResult:
        if not stl_path.exists():
            return SlicingResult(
                gcode_path=Path(""),
                gcode_hash="",
                errors=[f"STL file not found: {stl_path}"],
            )

        profile = profile_path or (
            Path(self.config.profile_path) if self.config.profile_path else None
        )

        try:
            slicer_info = await self._get_slicer()
        except FileNotFoundError as e:
            return SlicingResult(
                gcode_path=Path(""),
                gcode_hash="",
                errors=[str(e)],
            )

        profile = ProfileMapper.map_profile_path(profile, slicer_info.backend)

        stem = output_name or stl_path.stem
        gcode_path = self.output_dir / f"{stem}.gcode"

        cmd = self._build_slice_command(slicer_info, stl_path, gcode_path, profile)

        logger.info(
            f"Slicing {stl_path.name} with {slicer_info.backend.value} "
            f"({slicer_info.executable_path})"
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)

            stdout_text = stdout.decode(errors="replace") if stdout else ""
            stderr_text = stderr.decode(errors="replace") if stderr else ""

            if proc.returncode != 0:
                logger.error(f"Slicer failed (exit {proc.returncode}): {stderr_text}")
                return SlicingResult(
                    gcode_path=gcode_path,
                    gcode_hash="",
                    errors=[f"Slicer exited with code {proc.returncode}: {stderr_text.strip()}"],
                    slicer_backend=slicer_info.backend.value,
                    slicer_version=slicer_info.version,
                )

            if not gcode_path.exists():
                return SlicingResult(
                    gcode_path=gcode_path,
                    gcode_hash="",
                    errors=["Slicer completed but no G-code file was produced"],
                    slicer_backend=slicer_info.backend.value,
                    slicer_version=slicer_info.version,
                )

            # Extract metadata from generated G-code
            extractor = GcodeMetadataExtractor
            gcode_hash = extractor.compute_file_hash(gcode_path)
            first_layer_params = extractor.extract_first_layer_params(gcode_path)
            first_layer_bounds = extractor.extract_first_layer_bounds(gcode_path)
            metadata = extractor.extract_metadata(gcode_path)

            # Parse warnings from slicer output
            warnings = []
            for line in (stdout_text + stderr_text).splitlines():
                lower = line.lower()
                if "warning" in lower or "overhang" in lower or "bridge" in lower:
                    warnings.append(line.strip())

            estimated_time = None
            if "estimated_time" in metadata:
                estimated_time = timedelta(seconds=metadata["estimated_time"])

            logger.info(
                f"Slicing complete: {gcode_path.name} "
                f"(hash={gcode_hash[:12]}, warnings={len(warnings)})"
            )

            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash=gcode_hash,
                first_layer_params=first_layer_params,
                first_layer_bounds=first_layer_bounds,
                estimated_print_time=estimated_time,
                estimated_filament_mm=metadata.get("filament_used_mm"),
                slicer_backend=slicer_info.backend.value,
                slicer_version=slicer_info.version,
                warnings=warnings,
            )

        except asyncio.TimeoutError:
            logger.error(f"Slicer timed out after {timeout_seconds}s")
            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash="",
                errors=[f"Slicer timed out after {timeout_seconds}s"],
                slicer_backend=slicer_info.backend.value,
                slicer_version=slicer_info.version,
            )
        except FileNotFoundError:
            logger.error(f"Slicer not found: {slicer_info.executable_path}")
            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash="",
                errors=[f"Slicer executable not found: {slicer_info.executable_path}"],
                slicer_backend=slicer_info.backend.value,
                slicer_version=slicer_info.version,
            )
        except Exception as e:
            logger.error(f"Slicer error: {e}", exc_info=True)
            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash="",
                errors=[str(e)],
                slicer_backend=slicer_info.backend.value,
                slicer_version=slicer_info.version,
            )

    def format_summary(self, result: SlicingResult) -> str:
        lines = []

        if result.errors:
            lines.append("Slicing failed:")
            for err in result.errors:
                lines.append(f"  - {err}")
            return "\n".join(lines)

        lines.append(f"File: {result.gcode_path.name}")

        if result.slicer_backend:
            slicer_name = "OrcaSlicer" if result.slicer_backend == "orcaslicer" else "PrusaSlicer"
            version = f" v{result.slicer_version}" if result.slicer_version else ""
            lines.append(f"Slicer: {slicer_name}{version}")

        if result.estimated_print_time:
            total_min = int(result.estimated_print_time.total_seconds() / 60)
            hours, mins = divmod(total_min, 60)
            if hours:
                lines.append(f"Est. time: {hours}h {mins}min")
            else:
                lines.append(f"Est. time: {mins}min")

        if result.estimated_filament_mm:
            meters = result.estimated_filament_mm / 1000
            lines.append(f"Filament: {meters:.1f}m")

        if result.first_layer_params:
            p = result.first_layer_params
            lines.append(f"First layer: {p.layer_height}mm, {p.nozzle_temp}C/{p.bed_temp}C")

        if result.warnings:
            lines.append(f"Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                lines.append(f"  - {w}")

        return "\n".join(lines)
