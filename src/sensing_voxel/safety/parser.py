"""Strict G-code lexical parser used by the safety validator."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_WORD_RE = re.compile(rf"([A-Za-z])\s*({_NUMBER})")
_COMMAND_RE = re.compile(r"^([GMTgmt]\d+(?:\.\d+)?)")
_EXTENDED_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")

ALLOWED_COMMANDS = frozenset(
    {
        "G0", "G1", "G2", "G3", "G4", "G17", "G18", "G19", "G21", "G28",
        "G90", "G91", "G92", "M17", "M18", "M73", "M82", "M83", "M84",
        "M104", "M105", "M106", "M107", "M109", "M117", "M118", "M140",
        "M190", "M204", "M205", "M220", "M221", "M400", "M486",
    }
)
DANGEROUS_KLIPPER_COMMANDS = frozenset(
    {
        "ACTIVATE_EXTRUDER", "BED_MESH_CALIBRATE", "BED_TILT_CALIBRATE",
        "BLTOUCH_DEBUG", "CANCEL_PRINT", "DUMP_TMC", "FIRMWARE_RESTART", "FORCE_MOVE",
        "INIT_TMC", "MANUAL_STEPPER", "PAUSE", "PID_CALIBRATE", "PROBE_ACCURACY",
        "PROBE_CALIBRATE", "QUAD_GANTRY_LEVEL", "RESTART", "RESTORE_GCODE_STATE",
        "RESUME", "SAVE_CONFIG", "SAVE_GCODE_STATE", "SDCARD_PRINT_FILE",
        "SET_DUAL_CARRIAGE", "SET_EXTRUDER_ROTATION_DISTANCE", "SET_FAN_SPEED",
        "SET_GCODE_OFFSET", "SET_GCODE_VARIABLE", "SET_HEATER_TEMPERATURE",
        "SET_IDLE_TIMEOUT", "SET_INPUT_SHAPER", "SET_KINEMATIC_POSITION", "SET_PIN",
        "SET_PRESSURE_ADVANCE", "SET_SERVO", "SET_STEPPER_ENABLE", "SET_TMC_CURRENT",
        "SET_TMC_FIELD", "SET_VELOCITY_LIMIT", "SHAPER_CALIBRATE", "STEPPER_BUZZ",
        "SYNC_EXTRUDER_MOTION", "TURN_OFF_HEATERS", "TUNING_TOWER",
        "UPDATE_DELAYED_GCODE", "Z_TILT_ADJUST",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedLine:
    """One unambiguous executable line."""

    command: str
    params: dict[str, float]
    bare_axes: frozenset[str]


def strip_comments_and_checksum(raw_line: str) -> str:
    """Remove real comments and verify optional RepRap XOR checksum."""
    if "\x00" in raw_line or "{" in raw_line or "}" in raw_line:
        raise ValueError("control bytes and template expressions are forbidden")
    characters: list[str] = []
    in_parenthesis = False
    for character in raw_line:
        if character == ";" and not in_parenthesis:
            break
        if character == "(":
            if in_parenthesis:
                raise ValueError("nested parenthesis comments are ambiguous")
            in_parenthesis = True
        elif character == ")":
            if not in_parenthesis:
                raise ValueError("unmatched parenthesis comment")
            in_parenthesis = False
        elif not in_parenthesis:
            characters.append(character)
    if in_parenthesis:
        raise ValueError("unterminated parenthesis comment")
    code = "".join(characters).strip()
    if not code:
        return ""
    if "*" in code:
        before, checksum_text = code.rsplit("*", 1)
        if not checksum_text.strip().isdigit():
            raise ValueError("invalid checksum")
        checksum = 0
        for byte in before.encode("ascii", errors="strict"):
            checksum ^= byte
        if checksum != int(checksum_text.strip()):
            raise ValueError("checksum mismatch")
        code = before.strip()
    if re.match(r"^[Nn]\d+", code):
        code = re.sub(r"^[Nn]\d+\s*", "", code, count=1)
    return code.strip()


def parse_line(code: str, allowed_macros: tuple[str, ...] = ()) -> ParsedLine:
    """Parse a line or reject unknown commands, macros, and syntax."""
    match = _COMMAND_RE.match(code)
    if match is None:
        extended = _EXTENDED_RE.match(code)
        name = extended.group(1).upper() if extended else ""
        if name in DANGEROUS_KLIPPER_COMMANDS:
            raise ValueError(f"dangerous Klipper command '{name}' is forbidden")
        if name not in {macro.upper() for macro in allowed_macros}:
            raise ValueError(f"unknown command or macro '{name}'")
        if code[extended.end() :].strip() if extended else True:
            raise ValueError("allowed macros may not receive unvalidated parameters")
        return ParsedLine(name, {}, frozenset())

    command = match.group(1).upper()
    if command.endswith(".0"):
        command = command[:-2]
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"command '{command}' is not allowed")
    remainder = code[match.end() :].strip()
    if command in {"M117", "M118"}:
        if any(ord(character) < 32 and character != "\t" for character in remainder):
            raise ValueError("message contains control characters")
        return ParsedLine(command, {}, frozenset())

    params: dict[str, float] = {}
    bare_axes: set[str] = set()
    position = 0
    while position < len(remainder):
        while position < len(remainder) and remainder[position].isspace():
            position += 1
        if position >= len(remainder):
            break
        word = _WORD_RE.match(remainder, position)
        if word:
            key = word.group(1).upper()
            if key in params:
                raise ValueError(f"duplicate parameter '{key}'")
            value = float(word.group(2))
            if not math.isfinite(value):
                raise ValueError(f"non-finite parameter '{key}'")
            params[key] = value
            position = word.end()
            continue
        character = remainder[position].upper()
        if character in "XYZE" and command in {"G28", "M17", "M18", "M84"}:
            bare_axes.add(character)
            position += 1
            continue
        raise ValueError(f"ambiguous parameter syntax near '{remainder[position:]}'")
    return ParsedLine(command, params, frozenset(bare_axes))
