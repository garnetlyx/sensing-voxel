"""
Slicer pipeline for STL to G-code conversion via PrusaSlicer/OrcaSlicer CLI.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

from sensing_voxel.config import SlicerConfig
from sensing_voxel.core.models import SlicingResult, SlicingStatus
from sensing_voxel.slicer.metadata import GcodeMetadataExtractor

logger = logging.getLogger(__name__)


class SlicerPipeline:
    """Headless slicer integration via CLI subprocess."""

    def __init__(self, config: SlicerConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def slice_stl(
        self,
        stl_path: Path,
        profile_path: Path | None = None,
        output_name: str | None = None,
        timeout_seconds: int = 600,
    ) -> SlicingResult:
        """
        Slice an STL file to G-code using PrusaSlicer CLI.

        Args:
            stl_path: Path to input STL file
            profile_path: Override slicer profile (default from config)
            output_name: Override output filename (default from STL name)
            timeout_seconds: Subprocess timeout

        Returns:
            SlicingResult with gcode_path, metadata, and any warnings/errors
        """
        if not stl_path.exists():
            return SlicingResult(
                gcode_path=Path(""),
                gcode_hash="",
                errors=[f"STL file not found: {stl_path}"],
            )

        profile = profile_path or (
            Path(self.config.profile_path) if self.config.profile_path else None
        )

        stem = output_name or stl_path.stem
        gcode_path = self.output_dir / f"{stem}.gcode"

        cmd = [
            self.config.prusaslicer_path,
            "--export-gcode",
            str(stl_path),
            "--output", str(gcode_path),
        ]

        if profile and profile.exists():
            cmd.extend(["--load", str(profile)])

        logger.info(f"Slicing {stl_path.name} with {self.config.prusaslicer_path}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )

            stdout_text = stdout.decode(errors="replace") if stdout else ""
            stderr_text = stderr.decode(errors="replace") if stderr else ""

            if proc.returncode != 0:
                logger.error(f"Slicer failed (exit {proc.returncode}): {stderr_text}")
                return SlicingResult(
                    gcode_path=gcode_path,
                    gcode_hash="",
                    errors=[f"Slicer exited with code {proc.returncode}: {stderr_text.strip()}"],
                )

            if not gcode_path.exists():
                return SlicingResult(
                    gcode_path=gcode_path,
                    gcode_hash="",
                    errors=["Slicer completed but no G-code file was produced"],
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
                warnings=warnings,
            )

        except asyncio.TimeoutError:
            logger.error(f"Slicer timed out after {timeout_seconds}s")
            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash="",
                errors=[f"Slicer timed out after {timeout_seconds}s"],
            )
        except FileNotFoundError:
            logger.error(f"Slicer not found: {self.config.prusaslicer_path}")
            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash="",
                errors=[f"Slicer executable not found: {self.config.prusaslicer_path}"],
            )
        except Exception as e:
            logger.error(f"Slicer error: {e}", exc_info=True)
            return SlicingResult(
                gcode_path=gcode_path,
                gcode_hash="",
                errors=[str(e)],
            )

    def format_summary(self, result: SlicingResult) -> str:
        """Format slicing result as human-readable summary for Telegram."""
        lines = []

        if result.errors:
            lines.append("Slicing failed:")
            for err in result.errors:
                lines.append(f"  - {err}")
            return "\n".join(lines)

        lines.append(f"File: {result.gcode_path.name}")

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
            lines.append(
                f"First layer: {p.layer_height}mm, "
                f"{p.nozzle_temp}C/{p.bed_temp}C"
            )

        if result.warnings:
            lines.append(f"Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                lines.append(f"  - {w}")

        return "\n".join(lines)
