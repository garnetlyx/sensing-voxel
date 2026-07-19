"""Deterministic rendered preview generation for validated G-code artifacts."""

from __future__ import annotations

import hashlib
from html import escape

from sensing_voxel.safety.models import (
    Artifact,
    PreviewArtifact,
    ValidationError,
    ValidationResult,
    canonical_json_bytes,
    canonical_sha256,
)
from sensing_voxel.slicer.metadata import GcodeMetadataExtractor

_SVG_SIZE = 640
_SVG_MARGIN = 24
_MAX_RENDER_SEGMENTS = 20_000
_MAX_SVG_BYTES = 2_000_000


def _number(value: float) -> str:
    """Format finite validator-produced coordinates without platform drift."""
    formatted = format(value, ".6f").rstrip("0").rstrip(".")
    return "0" if formatted in {"", "-0"} else formatted


def _render_svg(artifact: Artifact, validation: ValidationResult) -> bytes:
    motions = tuple(motion for motion in validation.motions if motion.extrusion_delta > 0)
    if len(motions) > _MAX_RENDER_SEGMENTS:
        raise ValidationError("rendered preview exceeds the segment complexity limit")

    points = [point for motion in motions for point in (motion.start, motion.end)]
    if points:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
    else:
        min_x = max_x = min_y = max_y = 0.0

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    draw_size = _SVG_SIZE - (2 * _SVG_MARGIN)
    scale = min(draw_size / width, draw_size / height)
    offset_x = (_SVG_SIZE - (width * scale)) / 2
    offset_y = (_SVG_SIZE - (height * scale)) / 2

    def project(point: tuple[float, float, float]) -> tuple[str, str]:
        x = offset_x + ((point[0] - min_x) * scale)
        y = _SVG_SIZE - (offset_y + ((point[1] - min_y) * scale))
        return _number(x), _number(y)

    title = escape(f"Toolpath {artifact.sha256}", quote=True)
    description = escape(
        f"Validated extrusion toolpath with {len(motions)} rendered segments",
        quote=True,
    )
    lines = []
    for motion in motions:
        x1, y1 = project(motion.start)
        x2, y2 = project(motion.end)
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
        )

    body = "".join(lines)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_SIZE}" '
        f'height="{_SVG_SIZE}" viewBox="0 0 {_SVG_SIZE} {_SVG_SIZE}" '
        'role="img">'
        f"<title>{title}</title><desc>{description}</desc>"
        '<rect width="100%" height="100%" fill="#101418"/>'
        '<g fill="none" stroke="#55d6be" stroke-width="1.25" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</g>'
        "</svg>"
    ).encode()
    if len(svg) > _MAX_SVG_BYTES:
        raise ValidationError("rendered preview exceeds the byte-size limit")
    return svg


def generate_preview(
    artifact: Artifact, validation: ValidationResult
) -> PreviewArtifact:
    """Generate stable summary JSON and SVG from exact validated artifact bytes."""
    if not artifact.verify():
        raise ValidationError("artifact bytes do not match their content address")
    if not validation.valid:
        raise ValidationError("cannot preview G-code that failed validation")
    if validation.artifact_sha256 != artifact.sha256:
        raise ValidationError("validation does not belong to this artifact")
    if validation.result_sha256 != canonical_sha256(validation):
        raise ValidationError("validation result hash mismatch")

    metadata = GcodeMetadataExtractor.extract_metadata_bytes(artifact.content)
    rendered_content = _render_svg(artifact, validation)
    rendered_sha256 = hashlib.sha256(rendered_content).hexdigest()
    content = canonical_json_bytes(
        {
            "schema": "sensing-voxel.print-preview.v2",
            "artifact_sha256": artifact.sha256,
            "artifact_size_bytes": artifact.size_bytes,
            "validation_sha256": validation.result_sha256,
            "rendered_preview_sha256": rendered_sha256,
            "rendered_preview_media_type": "image/svg+xml",
            "command_count": validation.command_count,
            "line_count": validation.line_count,
            "toolpath_envelope": validation.toolpath_envelope,
            "first_layer_z": validation.first_layer_z,
            "first_layer_envelope": validation.first_layer_envelope,
            "metadata": metadata,
            "timeline": validation.timeline,
            "motions": validation.motions,
        }
    )
    return PreviewArtifact(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        rendered_content=rendered_content,
        rendered_sha256=rendered_sha256,
        artifact_sha256=artifact.sha256,
        validation_sha256=validation.result_sha256,
    )
