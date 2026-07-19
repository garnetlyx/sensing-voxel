"""Stateful motion and thermal validation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sensing_voxel.safety.models import Envelope, MotionRecord, SafetyLimits, ValidationIssue
from sensing_voxel.safety.parser import ParsedLine


@dataclass(slots=True)
class MachineState:
    """Subset of firmware parser and machine state required for proof."""

    absolute_xyz: bool = True
    absolute_e: bool = True
    plane: str = "XY"
    programmed_x: float = 0.0
    programmed_y: float = 0.0
    programmed_z: float = 0.0
    machine_x: float = 0.0
    machine_y: float = 0.0
    machine_z: float = 0.0
    extrusion: float = 0.0
    homed_x: bool = False
    homed_y: bool = False
    homed_z: bool = False
    hotend_target: float = 0.0
    saw_hotend_heat: bool = False

    def programmed(self, axis: str) -> float:
        return float(getattr(self, f"programmed_{axis.lower()}"))

    def machine(self, axis: str) -> float:
        return float(getattr(self, f"machine_{axis.lower()}"))

    def homed(self, axis: str) -> bool:
        return bool(getattr(self, f"homed_{axis.lower()}"))

    def set_axis(self, axis: str, programmed: float, machine: float) -> None:
        setattr(self, f"programmed_{axis.lower()}", programmed)
        setattr(self, f"machine_{axis.lower()}", machine)

    def set_homed(self, axis: str, value: bool) -> None:
        setattr(self, f"homed_{axis.lower()}", value)


def apply_command(
    parsed: ParsedLine,
    line: int,
    state: MachineState,
    limits: SafetyLimits,
    motions: list[MotionRecord],
    issues: list[ValidationIssue],
) -> None:
    """Apply one parsed command to state, recording all failures."""
    command, params = parsed.command, parsed.params
    if command == "G90":
        state.absolute_xyz = True
    elif command == "G91":
        state.absolute_xyz = False
    elif command == "M82":
        state.absolute_e = True
    elif command == "M83":
        state.absolute_e = False
    elif command in {"G17", "G18", "G19"}:
        state.plane = {"G17": "XY", "G18": "XZ", "G19": "YZ"}[command]
    elif command == "G28":
        _home(parsed, state, limits)
    elif command == "G92":
        _set_position(params, line, state, issues)
    elif command in {"M18", "M84"}:
        _disable(parsed, state)
    elif command in {"M104", "M109", "M140", "M190"}:
        _temperature(command, params, line, state, limits, issues)
    elif command in {"G0", "G1", "G2", "G3"}:
        _motion(command, params, line, state, limits, motions, issues)
    else:
        _validate_misc(command, params, line, limits, issues)


def _home(parsed: ParsedLine, state: MachineState, limits: SafetyLimits) -> None:
    axes = {key for key in parsed.params if key in "XYZ"} | (parsed.bare_axes & set("XYZ"))
    if not axes:
        axes = set("XYZ")
    origins = {"X": limits.min_x, "Y": limits.min_y, "Z": limits.min_z}
    for axis in axes:
        state.set_homed(axis, True)
        state.set_axis(axis, origins[axis], origins[axis])


def _disable(parsed: ParsedLine, state: MachineState) -> None:
    if "S" in parsed.params:
        return
    axes = {key for key in parsed.params if key in "XYZ"} | (parsed.bare_axes & set("XYZ"))
    if not axes:
        axes = set("XYZ")
    for axis in axes:
        state.set_homed(axis, False)


def _set_position(
    params: dict[str, float],
    line: int,
    state: MachineState,
    issues: list[ValidationIssue],
) -> None:
    for axis in "XYZ":
        if axis not in params:
            continue
        if not state.homed(axis):
            issues.append(ValidationIssue(line, "g92_unhomed", f"G92 {axis} before homing"))
            continue
        state.set_axis(axis, params[axis], state.machine(axis))
    if "E" in params:
        state.extrusion = params["E"]


def _temperature(
    command: str,
    params: dict[str, float],
    line: int,
    state: MachineState,
    limits: SafetyLimits,
    issues: list[ValidationIssue],
) -> None:
    target = params.get("S")
    if target is None:
        issues.append(ValidationIssue(line, "temperature_syntax", f"{command} requires S"))
        return
    maximum = limits.max_hotend_c if command in {"M104", "M109"} else limits.max_bed_c
    if target < 0 or target > maximum:
        issues.append(
            ValidationIssue(
                line,
                "temperature_bounds",
                f"{command} target {target:g} outside 0-{maximum:g}C",
            )
        )
        return
    if command in {"M104", "M109"}:
        state.hotend_target = target
        if target >= limits.min_extrude_c:
            state.saw_hotend_heat = True


def _motion(
    command: str,
    params: dict[str, float],
    line: int,
    state: MachineState,
    limits: SafetyLimits,
    motions: list[MotionRecord],
    issues: list[ValidationIssue],
) -> None:
    for axis in (axis for axis in "XYZ" if axis in params):
        if not state.homed(axis):
            issues.append(
                ValidationIssue(
                    line, "motion_before_homing", f"{command} moves {axis} before homing"
                )
            )
    feedrate = params.get("F")
    if feedrate is not None and (feedrate <= 0 or feedrate > limits.max_feedrate_mm_min):
        issues.append(ValidationIssue(line, "feedrate_bounds", "feedrate outside profile"))

    start = (state.machine_x, state.machine_y, state.machine_z)
    programmed: dict[str, float] = {}
    machine: dict[str, float] = {}
    for axis in "XYZ":
        value = params.get(axis)
        if value is None:
            programmed[axis] = state.programmed(axis)
            machine[axis] = state.machine(axis)
        elif state.absolute_xyz:
            offset = state.machine(axis) - state.programmed(axis)
            programmed[axis] = value
            machine[axis] = value + offset
        else:
            programmed[axis] = state.programmed(axis) + value
            machine[axis] = state.machine(axis) + value

    extrusion_delta = _extrusion_delta(params, state)
    if extrusion_delta > 0 and state.hotend_target < limits.min_extrude_c:
        issues.append(
            ValidationIssue(line, "cold_extrusion", "positive extrusion without hotend target")
        )
    end = (machine["X"], machine["Y"], machine["Z"])
    envelope = _arc_envelope(command, params, start, end, state.plane, limits, line, issues)
    _check_envelope(envelope, limits, line, issues)
    motions.append(MotionRecord(line, command, start, end, envelope, extrusion_delta))
    for axis in "XYZ":
        state.set_axis(axis, programmed[axis], machine[axis])


def _extrusion_delta(params: dict[str, float], state: MachineState) -> float:
    if "E" not in params:
        return 0.0
    if state.absolute_e:
        delta = params["E"] - state.extrusion
        state.extrusion = params["E"]
        return delta
    state.extrusion += params["E"]
    return params["E"]


def _arc_envelope(
    command: str,
    params: dict[str, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    plane: str,
    limits: SafetyLimits,
    line: int,
    issues: list[ValidationIssue],
) -> Envelope:
    if command not in {"G2", "G3"}:
        return _line_envelope(start, end)
    plane_spec = {
        "XY": (0, 1, "I", "J"),
        "XZ": (0, 2, "I", "K"),
        "YZ": (1, 2, "J", "K"),
    }
    first, second, first_offset, second_offset = plane_spec[plane]
    if "R" in params:
        if limits.firmware.lower() == "klipper":
            issues.append(ValidationIssue(line, "arc_ambiguity", "Klipper R arcs are unsupported"))
        else:
            issues.append(ValidationIssue(line, "arc_ambiguity", "R arcs are not proven safe"))
        return _line_envelope(start, end)
    wrong_offsets = {"I", "J", "K"} - {first_offset, second_offset}
    if any(offset in params for offset in wrong_offsets):
        issues.append(ValidationIssue(line, "arc_plane", "arc offset does not match active plane"))
        return _line_envelope(start, end)
    if first_offset not in params and second_offset not in params:
        issues.append(ValidationIssue(line, "arc_ambiguity", "arc center offsets are required"))
        return _line_envelope(start, end)

    center_first = start[first] + params.get(first_offset, 0.0)
    center_second = start[second] + params.get(second_offset, 0.0)
    radius_start = math.hypot(start[first] - center_first, start[second] - center_second)
    radius_end = math.hypot(end[first] - center_first, end[second] - center_second)
    tolerance = max(0.005, min(0.5, radius_start * 0.001))
    if radius_start <= 1e-9 or abs(radius_start - radius_end) > tolerance:
        issues.append(ValidationIssue(line, "arc_geometry", "arc radii are inconsistent"))
        return _line_envelope(start, end)

    start_angle = math.atan2(start[second] - center_second, start[first] - center_first)
    end_angle = math.atan2(end[second] - center_second, end[first] - center_first)
    sweep = _arc_sweep(start_angle, end_angle, clockwise=command == "G2")
    points = [start, end]
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        if _angle_in_sweep(angle, start_angle, sweep):
            fraction = _angle_fraction(angle, start_angle, sweep)
            point = list(start)
            point[first] = center_first + radius_start * math.cos(angle)
            point[second] = center_second + radius_start * math.sin(angle)
            remaining = ({0, 1, 2} - {first, second}).pop()
            point[remaining] = start[remaining] + fraction * (end[remaining] - start[remaining])
            points.append((point[0], point[1], point[2]))
    return Envelope(
        min(point[0] for point in points), max(point[0] for point in points),
        min(point[1] for point in points), max(point[1] for point in points),
        min(point[2] for point in points), max(point[2] for point in points),
    )


def _arc_sweep(start: float, end: float, clockwise: bool) -> float:
    if clockwise:
        sweep = -((start - end) % math.tau)
        return sweep if sweep != 0 else -math.tau
    sweep = (end - start) % math.tau
    return sweep if sweep != 0 else math.tau


def _angle_in_sweep(angle: float, start: float, sweep: float) -> bool:
    if sweep > 0:
        return ((angle - start) % math.tau) <= sweep + 1e-12
    return ((start - angle) % math.tau) <= -sweep + 1e-12


def _angle_fraction(angle: float, start: float, sweep: float) -> float:
    distance = (angle - start) % math.tau if sweep > 0 else (start - angle) % math.tau
    return distance / abs(sweep)


def _line_envelope(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> Envelope:
    return Envelope(
        min(start[0], end[0]), max(start[0], end[0]),
        min(start[1], end[1]), max(start[1], end[1]),
        min(start[2], end[2]), max(start[2], end[2]),
    )


def _check_envelope(
    envelope: Envelope,
    limits: SafetyLimits,
    line: int,
    issues: list[ValidationIssue],
) -> None:
    comparisons = (
        ("X", envelope.min_x, envelope.max_x, limits.min_x, limits.max_x),
        ("Y", envelope.min_y, envelope.max_y, limits.min_y, limits.max_y),
        ("Z", envelope.min_z, envelope.max_z, limits.min_z, limits.max_z),
    )
    for axis, low, high, allowed_low, allowed_high in comparisons:
        if low is not None and high is not None and (low < allowed_low or high > allowed_high):
            issues.append(
                ValidationIssue(line, "motion_bounds", f"{axis} envelope outside profile bounds")
            )


def _validate_misc(
    command: str,
    params: dict[str, float],
    line: int,
    limits: SafetyLimits,
    issues: list[ValidationIssue],
) -> None:
    if command == "G21" and params:
        issues.append(ValidationIssue(line, "unexpected_parameter", "G21 takes no parameters"))
    elif command == "G4":
        dwell = params.get("P", params.get("S"))
        if dwell is None or dwell < 0 or ("P" in params and dwell > 60_000):
            issues.append(ValidationIssue(line, "dwell_bounds", "invalid or excessive dwell"))
    elif command in {"M106"} and not (0 <= params.get("S", 255) <= 255):
        issues.append(ValidationIssue(line, "fan_bounds", "fan speed must be 0-255"))
    elif command in {"M220", "M221"} and not (1 <= params.get("S", 0) <= 200):
        issues.append(ValidationIssue(line, "override_bounds", "override must be 1-200 percent"))
    elif command == "M204":
        values = [value for key, value in params.items() if key in {"S", "P", "T"}]
        if not values or any(value <= 0 or value > limits.max_feedrate_mm_min for value in values):
            issues.append(ValidationIssue(line, "acceleration_bounds", "invalid acceleration"))
