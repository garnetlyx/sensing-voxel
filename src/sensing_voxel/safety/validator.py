"""Whole-file deterministic G-code validation."""

from __future__ import annotations

from dataclasses import replace

from sensing_voxel.safety.models import (
    Artifact,
    Envelope,
    MotionRecord,
    SafetyLimits,
    TimelineEntry,
    ValidationIssue,
    ValidationResult,
    canonical_sha256,
)
from sensing_voxel.safety.motion import MachineState, apply_command
from sensing_voxel.safety.parser import parse_line, strip_comments_and_checksum

_PARAMETER_ALLOWLISTS: dict[str, frozenset[str]] = {
    "G0": frozenset("XYZEF"),
    "G1": frozenset("XYZEF"),
    "G2": frozenset("XYZEFIJKR"),
    "G3": frozenset("XYZEFIJKR"),
    "G4": frozenset("PS"),
    "G17": frozenset(),
    "G18": frozenset(),
    "G19": frozenset(),
    "G21": frozenset(),
    "G28": frozenset("XYZ"),
    "G90": frozenset(),
    "G91": frozenset(),
    "G92": frozenset("XYZE"),
    "M17": frozenset("XYZ"),
    "M18": frozenset("XYZS"),
    "M73": frozenset("PR"),
    "M82": frozenset(),
    "M83": frozenset(),
    "M84": frozenset("XYZS"),
    "M104": frozenset("ST"),
    "M105": frozenset(),
    "M106": frozenset("SP"),
    "M107": frozenset(),
    "M109": frozenset("ST"),
    "M117": frozenset(),
    "M118": frozenset(),
    "M140": frozenset("ST"),
    "M190": frozenset("ST"),
    "M204": frozenset("SPT"),
    "M205": frozenset("XYZEJST"),
    "M220": frozenset("S"),
    "M221": frozenset("S"),
    "M400": frozenset(),
    "M486": frozenset("PST"),
}


class GcodeValidator:
    """Validate every executable line while tracking machine state."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    def validate(self, artifact: Artifact) -> ValidationResult:
        issues: list[ValidationIssue] = []
        timeline: list[TimelineEntry] = []
        motions: list[MotionRecord] = []
        state = MachineState()
        command_count = 0
        saw_homing = False
        command_limit_reported = False

        if not artifact.verify():
            issues.append(ValidationIssue(None, "artifact_integrity", "artifact bytes do not match hash"))

        try:
            text = artifact.content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            issues.append(ValidationIssue(exc.start + 1, "encoding", "G-code must be valid UTF-8"))
            text = ""

        lines = text.splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            try:
                code = strip_comments_and_checksum(raw_line)
                if not code:
                    continue
                parsed = parse_line(code, self.limits.allowed_macros)
                command_count += 1
                if command_count > self.limits.max_commands and not command_limit_reported:
                    issues.append(
                        ValidationIssue(
                            line_number,
                            "command_limit",
                            "command count exceeds profile",
                        )
                    )
                    command_limit_reported = True
                allowed_parameters = _PARAMETER_ALLOWLISTS.get(parsed.command, frozenset())
                unexpected = (set(parsed.params) | set(parsed.bare_axes)) - allowed_parameters
                if unexpected:
                    names = ", ".join(sorted(unexpected))
                    raise ValueError(f"unexpected parameter(s) for {parsed.command}: {names}")
                parameters: list[tuple[str, float | str]] = sorted(
                    parsed.params.items()
                )
                parameters.extend((axis, "bare") for axis in sorted(parsed.bare_axes))
                timeline.append(
                    TimelineEntry(line_number, parsed.command, tuple(parameters))
                )
                if parsed.command == "G28":
                    saw_homing = True
                apply_command(parsed, line_number, state, self.limits, motions, issues)
            except (UnicodeEncodeError, ValueError) as exc:
                issues.append(ValidationIssue(line_number, "parse_error", str(exc)))

        if not timeline:
            issues.append(ValidationIssue(None, "empty_program", "no executable G-code commands"))
        if not saw_homing:
            issues.append(
                ValidationIssue(None, "missing_homing", "no executable G28 command")
            )
        if not state.saw_hotend_heat:
            issues.append(
                ValidationIssue(
                    None,
                    "missing_hotend_heat",
                    "no executable hotend heating command reaches extrusion temperature",
                )
            )

        toolpath = _merge_envelopes(tuple(motion.envelope for motion in motions))
        first_layer_z, first_layer = _first_layer(tuple(motions))
        result = ValidationResult(
            artifact_sha256=artifact.sha256,
            valid=not issues,
            command_count=command_count,
            line_count=len(lines),
            toolpath_envelope=toolpath,
            first_layer_envelope=first_layer,
            first_layer_z=first_layer_z,
            timeline=tuple(timeline),
            motions=tuple(motions),
            issues=tuple(issues),
            result_sha256="",
        )
        return replace(result, result_sha256=canonical_sha256(result))


def _first_layer(motions: tuple[MotionRecord, ...]) -> tuple[float | None, Envelope]:
    extrusion_motions = [motion for motion in motions if motion.extrusion_delta > 0]
    if not extrusion_motions:
        return None, Envelope()
    first_z = min(min(motion.start[2], motion.end[2]) for motion in extrusion_motions)
    tolerance = 1e-6
    envelopes = tuple(
        motion.envelope
        for motion in extrusion_motions
        if min(motion.start[2], motion.end[2]) <= first_z + tolerance
    )
    return first_z, _merge_envelopes(envelopes)


def _merge_envelopes(envelopes: tuple[Envelope, ...]) -> Envelope:
    if not envelopes:
        return Envelope()

    def minimum(name: str) -> float | None:
        values = [getattr(envelope, name) for envelope in envelopes]
        concrete = [value for value in values if value is not None]
        return min(concrete) if concrete else None

    def maximum(name: str) -> float | None:
        values = [getattr(envelope, name) for envelope in envelopes]
        concrete = [value for value in values if value is not None]
        return max(concrete) if concrete else None

    return Envelope(
        min_x=minimum("min_x"),
        max_x=maximum("max_x"),
        min_y=minimum("min_y"),
        max_y=maximum("max_y"),
        min_z=minimum("min_z"),
        max_z=maximum("max_z"),
    )
