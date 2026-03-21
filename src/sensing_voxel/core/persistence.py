"""
State persistence with atomic snapshots

Adapted from OpenClaw's src/config/sessions/store.ts pattern
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from filelock import SoftFileLock

from sensing_voxel.core.models import AgentState
from sensing_voxel.core.state_machine import AgentStateMachine

logger = logging.getLogger(__name__)


class StatePersistenceError(Exception):
    """Raised when state persistence fails"""

    pass


class StatePersistence:
    """Atomic state persistence to JSON file"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._lock_file: Path | None = None
        self._lock_instance: SoftFileLock | None = None
        self._async_lock = asyncio.Lock()

        # Ensure parent directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _acquire_lock(self):
        """Acquire file lock"""
        if self._lock_instance is not None and self._lock_instance.is_locked:
            return

        self._lock_file = self.state_file.with_suffix(".lock")

        try:
            self._lock_instance = SoftFileLock(self._lock_file, timeout=5)
            self._lock_instance.acquire()
            logger.debug("Acquired state file lock")
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            raise

    def _release_lock(self):
        """Release file lock"""
        if self._lock_instance is not None:
            try:
                self._lock_instance.release()
                self._lock_instance = None
                if self._lock_file and self._lock_file.exists():
                    self._lock_file.unlink()
            except Exception as e:
                logger.error(f"Error releasing lock: {e}")

    def _to_dict(self, state_machine: AgentStateMachine) -> dict:
        """Convert state machine to dict for serialization"""
        job_context_data = None
        if state_machine.job_context:
            jc = state_machine.job_context
            job_context_data = {
                "job_id": jc.job_id,
                "job_file": jc.job_file,
                "uploaded_at": jc.uploaded_at.isoformat(),
                "expires_at": (jc.expires_at.isoformat() if jc.expires_at else None),
                "decision_token": jc.decision_token,
                "decision_consumed": jc.decision_consumed,
                "decision_action": (jc.decision_action.name if jc.decision_action else None),
                "decision_consumed_at": (
                    jc.decision_consumed_at.isoformat() if jc.decision_consumed_at else None
                ),
                "gcode_path": jc.gcode_path,
            }

        return {
            "schema_version": 1,
            "current_state": state_machine.current_state.name,
            "job_context": job_context_data,
            "last_error_summary": state_machine.last_error_summary,
            "z_offset_dirty": state_machine.z_offset_dirty,
            "last_known_z_offset_baseline": state_machine.last_known_z_offset_baseline,
            "moonraker_compat_mode": state_machine.moonraker_compat_mode,
            "external_print": state_machine.external_print,
            "timestamp": datetime.now().isoformat(),
        }

    def _from_dict(self, data: dict) -> AgentStateMachine:
        """Create state machine from dict"""
        from sensing_voxel.core.models import DecisionAction, JobContext

        version = data.get("schema_version", 0)
        if version != 1:
            logger.warning(f"Loading state with schema version {version}, expected 1")

        # Validate required fields
        if "current_state" not in data:
            logger.error("Missing required field 'current_state' in state data")
            raise ValueError("Missing required field: current_state")

        state_machine = AgentStateMachine()
        try:
            state_machine.current_state = AgentState[data["current_state"]]
        except KeyError as err:
            logger.error(f"Invalid state value: {data['current_state']}")
            raise ValueError(f"Invalid state value: {data['current_state']}") from err

        job_data = data.get("job_context")
        if job_data:
            state_machine.job_context = JobContext(
                job_id=job_data["job_id"],
                job_file=job_data["job_file"],
                uploaded_at=datetime.fromisoformat(job_data["uploaded_at"]),
                expires_at=(
                    datetime.fromisoformat(job_data["expires_at"])
                    if job_data["expires_at"]
                    else None
                ),
                decision_token=job_data["decision_token"],
                decision_consumed=job_data["decision_consumed"],
                decision_action=(
                    DecisionAction[job_data["decision_action"]]
                    if job_data["decision_action"]
                    else None
                ),
                decision_consumed_at=(
                    datetime.fromisoformat(job_data["decision_consumed_at"])
                    if job_data["decision_consumed_at"]
                    else None
                ),
                gcode_path=job_data["gcode_path"],
            )

        state_machine.last_error_summary = data.get("last_error_summary")
        state_machine.z_offset_dirty = data.get("z_offset_dirty", False)
        state_machine.last_known_z_offset_baseline = data.get("last_known_z_offset_baseline")
        state_machine.moonraker_compat_mode = data.get("moonraker_compat_mode", False)
        state_machine.external_print = data.get("external_print", False)

        return state_machine

    def save(self, state_machine: AgentStateMachine):
        """Atomically save state machine to file"""
        tmp_path = None
        self._acquire_lock()
        try:
            state_dict = self._to_dict(state_machine)

            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.state_file.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(state_dict, tmp_file, indent=2)
                tmp_path = Path(tmp_file.name)

            # Atomic rename
            tmp_path.rename(self.state_file)
            logger.info(f"State saved to {self.state_file}")

        except Exception as e:
            logger.error(f"Error saving state: {e}")
            # Clean up orphaned temp file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise StatePersistenceError(f"Failed to save state: {e}") from e
        finally:
            self._release_lock()

    async def save_async(self, state_machine: AgentStateMachine):
        """Async-safe save with serialization lock to prevent concurrent writes"""
        async with self._async_lock:
            self.save(state_machine)

    def load(self) -> AgentStateMachine | None:
        """Load state machine from file"""
        if not self.state_file.exists():
            logger.info("No state file found, starting fresh")
            return None

        self._acquire_lock()
        try:
            with open(self.state_file) as f:
                state_dict = json.load(f)

            state_machine = self._from_dict(state_dict)
            logger.info(f"State loaded from {self.state_file}")
            return state_machine

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return None
        finally:
            self._release_lock()

    def reset_to_idle(self, state_machine: AgentStateMachine):
        """Reset state machine to IDLE, preserving safety fields"""
        # Clear job-related fields
        state_machine.job_context = None
        state_machine.last_error_summary = None
        state_machine.external_print = False

        # Set state to IDLE
        state_machine.current_state = AgentState.IDLE

        # Preserve safety fields
        # z_offset_dirty and last_known_z_offset_baseline are preserved

        self.save(state_machine)
