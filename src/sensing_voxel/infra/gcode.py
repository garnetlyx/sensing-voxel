"""
G-code validation utilities for 3D printer command safety.

Provides centralized G-code validation with security checks for:
- Blocked dangerous G-code prefixes (emergency stop, power control, etc.)
- Temperature bounds for heating commands
- Axis movement bounds
- G-code command format validation
"""

import re

# Blocked G-code prefixes that could be dangerous or disruptive
BLOCKED_GCODE_PREFIXES = ("M112", "M410", "M80", "M81", "M23", "\n", "\r")

# File-print commands and conventional macro aliases must only be reached through
# PrintSafetyService.  Generic raw-command approval is intentionally insufficient.
PRINT_START_GCODE_COMMANDS = frozenset({"M23", "M24", "M32", "SDCARD_PRINT_FILE"})
_PRINT_START_ALIAS_RE = re.compile(
    r"(?:^|_)(?:PRINT_START|START_PRINT|SDCARD_PRINT_FILE|RESUME)(?:_|$)"
)
PRINT_START_SAFETY_FLOW = "the PrintSafetyService start_print flow"

# Temperature bounds for heating commands (in Celsius)
GCODE_TEMP_BOUNDS = {
    "M104": (0.0, 300.0),  # Set hotend temp
    "M109": (0.0, 300.0),  # Set hotend temp and wait
    "M140": (0.0, 120.0),  # Set bed temp
    "M190": (0.0, 120.0),  # Set bed temp and wait
}

# Axis movement bounds (in mm)
GCODE_AXIS_BOUNDS = {
    "X": (0.0, 350.0),
    "Y": (0.0, 350.0),
    "Z": (0.0, 400.0),
    "E": (-10.0, 100.0),
}

# G-code token validation regex patterns
_GCODE_CMD_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_GCODE_LETTER_NUM_RE = re.compile(r"^[A-Z][-+]?(?:\d+(?:\.\d+)?|\.\d+)$")
_GCODE_KEY_VALUE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=[-+]?(?:\d+(?:\.\d+)?|\.\d+)$")
_GCODE_NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)$")


def _strip_gcode_comment(command: str) -> str:
    """Strip semicolon comments while preserving the executable command text."""
    return command.split(";", 1)[0].strip()


def is_print_start_command(command: str) -> bool:
    """Return whether one lexical command can start or resume a file print."""
    executable = _strip_gcode_comment(command)
    if not executable or "\n" in executable or "\r" in executable:
        return False
    token = executable.split(maxsplit=1)[0].upper()
    return token in PRINT_START_GCODE_COMMANDS or _PRINT_START_ALIAS_RE.search(token) is not None


def validate_macro_command(command: str) -> str | None:
    """Validate one macro invocation, including optional parameters."""
    if "\n" in command or "\r" in command:
        return "Only one macro command is allowed"
    executable = _strip_gcode_comment(command)
    if not executable:
        return "Macro name cannot be empty"

    parts = executable.split()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", parts[0]):
        return "Invalid macro name"
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", part) for part in parts[1:]):
        return "Macro parameters must use KEY=VALUE syntax"
    return None


def print_start_block_message() -> str:
    return f"File print start/resume is blocked in generic tools; use {PRINT_START_SAFETY_FLOW}"



def _extract_numeric_gcode_params(tokens: list[str]) -> dict[str, float]:
    """Extract numeric parameters from G-code tokens.

    Supports formats like:
    - X10.5 (letter + number)
    - KEY=VALUE (key-value pairs)
    """
    params: dict[str, float] = {}
    for token in tokens[1:]:
        upper = token.upper()
        if "=" in upper:
            key, value = upper.split("=", 1)
            if _GCODE_NUMBER_RE.fullmatch(value):
                params[key] = float(value)
            continue
        if len(upper) >= 2 and upper[0].isalpha() and _GCODE_NUMBER_RE.fullmatch(upper[1:]):
            params[upper[0]] = float(upper[1:])
    return params


def validate_gcode_command(command: str) -> str | None:
    """Validate a G-code command for safety and correctness.

    Args:
        command: Raw G-code command string to validate

    Returns:
        Error message string if validation fails, None if valid

    Checks performed:
    1. Single line check (no newlines)
    2. Non-empty after stripping comments
    3. Blocked dangerous prefixes (M112 emergency stop, etc.)
    4. Valid command token format
    5. Valid parameter token format
    6. Temperature bounds for heating commands
    7. Axis bounds for movement commands
    """
    if "\n" in command or "\r" in command:
        return "G-code must be a single line. Newlines are not allowed."

    normalized = _strip_gcode_comment(command)
    if not normalized:
        return "command is required"

    upper = normalized.upper()
    for prefix in BLOCKED_GCODE_PREFIXES:
        if prefix in {"\n", "\r"}:
            continue
        if re.match(rf"^{re.escape(prefix)}(?:\s|$)", upper):
            return f"G-code '{prefix}' is blocked. Use emergency_cancel instead."

    tokens = upper.split()
    if not _GCODE_CMD_RE.fullmatch(tokens[0]):
        return "Invalid G-code structure: bad command token"

    for token in tokens[1:]:
        if (
            _GCODE_LETTER_NUM_RE.fullmatch(token)
            or _GCODE_KEY_VALUE_RE.fullmatch(token)
            or _GCODE_CMD_RE.fullmatch(token)
            or _GCODE_NUMBER_RE.fullmatch(token)
        ):
            continue
        return f"Invalid G-code structure: bad parameter token '{token}'"

    command_code = tokens[0]
    params = _extract_numeric_gcode_params(tokens)

    if command_code in GCODE_TEMP_BOUNDS:
        if "S" not in params:
            return f"{command_code} requires S temperature parameter"
        min_temp, max_temp = GCODE_TEMP_BOUNDS[command_code]
        temp = params["S"]
        if temp < min_temp or temp > max_temp:
            return f"{command_code} S must be {int(min_temp)}-{int(max_temp)}C, got {temp}"

    if command_code in {"G0", "G1"}:
        for axis, (lower, upper_bound) in GCODE_AXIS_BOUNDS.items():
            if axis not in params:
                continue
            value = params[axis]
            if value < lower or value > upper_bound:
                return f"{command_code} {axis} must be {lower:g}-{upper_bound:g}mm, got {value}"

    return None
