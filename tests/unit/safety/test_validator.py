from __future__ import annotations

import hashlib
import json

import pytest

from sensing_voxel.safety import Artifact, GcodeValidator, SafetyLimits, ValidationError
from sensing_voxel.safety.preview import generate_preview


def validate(gcode: str, *, limits: SafetyLimits | None = None):
    return GcodeValidator(limits).validate(Artifact.from_bytes(gcode.encode()))


def issue_codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_motion_without_prior_homing_is_blocked() -> None:
    result = validate("M104 S200\nG1 X10 F1200\nG28\n")

    assert not result.valid
    assert "motion_before_homing" in issue_codes(result)


@pytest.mark.parametrize(
    "gcode",
    [
        "; G28\nM104 S200\nG1 X10\n",
        "(G28)\nM104 S200\nG1 X10\n",
        "M117 G28 M104 S200\nG1 X10\n",
    ],
)
def test_comments_and_message_strings_cannot_fake_homing(gcode: str) -> None:
    result = validate(gcode)

    assert not result.valid
    assert "missing_homing" in issue_codes(result)
    assert "motion_before_homing" in issue_codes(result)


@pytest.mark.parametrize(
    "gcode",
    [
        "G28\n; M104 S200\nG1 X10\n",
        "G28\n(M109 S200)\nG1 X10\n",
        "G28\nM117 M104 S200\nG1 X10\n",
    ],
)
def test_comments_and_message_strings_cannot_fake_heating(gcode: str) -> None:
    result = validate(gcode)

    assert not result.valid
    assert "missing_hotend_heat" in issue_codes(result)


def test_validator_parses_unsafe_content_after_first_5000_characters() -> None:
    prefix = ";" + ("padding" * 800) + "\nG28\nM104 S200\n"
    result = validate(prefix + "M112\n")

    assert not result.valid
    assert any("M112" in issue.message for issue in result.issues)
    assert result.line_count == 4


def test_command_limit_does_not_stop_later_unsafe_parsing() -> None:
    limits = SafetyLimits(max_commands=3)
    result = validate(
        "G28\nM104 S200\nG90\nG1 X1\nSAVE_CONFIG\n",
        limits=limits,
    )

    assert not result.valid
    assert "command_limit" in issue_codes(result)
    assert any("SAVE_CONFIG" in issue.message for issue in result.issues)
    assert result.command_count == 4


@pytest.mark.parametrize(
    "line, expected_code",
    [
        ("G1 X1 X2", "parse_error"),
        ("G2 X10 Y10 R5", "arc_ambiguity"),
        ("G2 X10 Y10", "arc_ambiguity"),
        ("G2 X10 Y10 K2", "arc_plane"),
        ("UNREVIEWED_MACRO", "parse_error"),
    ],
)
def test_unknown_or_ambiguous_commands_fail_closed(line: str, expected_code: str) -> None:
    result = validate(f"G28\nM104 S200\n{line}\n")

    assert not result.valid
    assert expected_code in issue_codes(result)


def test_relative_motion_and_extrusion_are_tracked() -> None:
    result = validate(
        "G28\nM104 S200\nG90\nM83\n"
        "G1 X10 Y10 Z0.2 F1200\n"
        "G91\nG1 X5 E1\n"
    )

    assert result.valid, result.issues
    assert result.motions[-1].start == (10.0, 10.0, 0.2)
    assert result.motions[-1].end == (15.0, 10.0, 0.2)
    assert result.motions[-1].extrusion_delta == 1.0
    assert result.first_layer_z == 0.2


def test_partial_homing_does_not_authorize_unhomed_axis_motion() -> None:
    result = validate("G28 X Y\nM104 S200\nG1 X10 Y10 Z0.2 F1200\n")

    assert not result.valid
    assert "motion_before_homing" in issue_codes(result)
    assert any("moves Z before homing" in issue.message for issue in result.issues)


def test_absolute_extrusion_mode_tracks_delta_across_g92_reset() -> None:
    result = validate(
        "G28\nM104 S200\nG90\nM82\n"
        "G1 X10 Y10 Z0.2 E5 F1200\n"
        "G92 E0\n"
        "G1 X20 E1\n"
    )

    assert result.valid, result.issues
    assert result.motions[-1].extrusion_delta == 1.0


def test_g92_does_not_hide_machine_bounds() -> None:
    result = validate("G28\nM104 S200\nG92 X300\nG1 X700\n")

    assert not result.valid
    assert "motion_bounds" in issue_codes(result)


def test_motor_disable_invalidates_homing() -> None:
    result = validate("G28\nM104 S200\nM84 X\nG1 X1\n")

    assert not result.valid
    assert "motion_before_homing" in issue_codes(result)


def test_preview_is_deterministic_and_bound_to_exact_artifact() -> None:
    artifact = Artifact.from_bytes(b"G28\nM104 S200\nG1 X10 Y20 Z0.2\n")
    validation = GcodeValidator().validate(artifact)

    first = generate_preview(artifact, validation)
    second = generate_preview(artifact, validation)
    payload = json.loads(first.content)

    assert first == second
    assert first.sha256 == hashlib.sha256(first.content).hexdigest()
    assert payload["artifact_sha256"] == artifact.sha256
    assert payload["validation_sha256"] == validation.result_sha256
    assert payload["toolpath_envelope"]["max_x"] == 10.0

    changed = Artifact.from_bytes(artifact.content + b"; changed\n")
    with pytest.raises(ValidationError, match="does not belong"):
        generate_preview(changed, validation)
