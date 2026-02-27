"""Dataclass to dict serializers for API responses"""

from datetime import datetime

from sensing_voxel.core.models import (
    AgentState,
    JobContext,
    NormalizedPrinterState,
    PrintProgress,
    TelemetryData,
)
from sensing_voxel.core.state_machine import AgentStateMachine


def serialize_state_machine(sm: AgentStateMachine) -> dict:
    """Serialize AgentStateMachine to API-friendly dict"""
    return {
        "agent_state": sm.current_state.name,
        "has_job": sm.job_context is not None,
        "external_print": sm.external_print,
        "z_offset_dirty": sm.z_offset_dirty,
        "last_error": sm.last_error_summary,
    }


def serialize_job_context(job: JobContext | None) -> dict | None:
    """Serialize JobContext to API-friendly dict"""
    if job is None:
        return None

    return {
        "job_id": job.job_id,
        "job_file": job.job_file,
        "uploaded_at": job.uploaded_at.isoformat(),
        "expires_at": job.expires_at.isoformat() if job.expires_at else None,
        "decision_consumed": job.decision_consumed,
        "decision_action": job.decision_action.name if job.decision_action else None,
        "gcode_path": job.gcode_path,
        "is_external": job.is_external,
        "external_monitoring": job.external_monitoring,
    }


def serialize_telemetry(telemetry: TelemetryData | None) -> dict | None:
    """Serialize TelemetryData to API-friendly dict"""
    if telemetry is None:
        return None

    return {
        "timestamp": telemetry.timestamp.isoformat(),
        "temperatures": telemetry.temperatures,
        "position": telemetry.position,
        "velocity": telemetry.velocity,
        "metadata": telemetry.metadata,
    }


def serialize_print_progress(progress: PrintProgress | None) -> dict | None:
    """Serialize PrintProgress to API-friendly dict"""
    if progress is None:
        return None

    return {
        "state": progress.state.name,
        "filename": progress.filename,
        "progress": progress.progress,
        "print_time": progress.print_time,
        "current_layer": progress.current_layer,
        "total_layers": progress.total_layers,
        "nozzle_temp": progress.nozzle_temp,
        "nozzle_target": progress.nozzle_target,
        "bed_temp": progress.bed_temp,
        "bed_target": progress.bed_target,
    }


def serialize_alert(alert: dict) -> dict:
    """Serialize an alert entry"""
    return {
        "type": alert.get("type", "info"),
        "message": alert.get("message", ""),
        "timestamp": alert.get("timestamp", datetime.now().isoformat()),
        "data": alert.get("data"),
    }
