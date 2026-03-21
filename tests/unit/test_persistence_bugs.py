"""
BUG REPRODUCTION: StatePersistence issues

These tests expose critical bugs in StatePersistence:
1. Lock not released if exception during save
2. exit(0) in signal handler kills entire process
3. No cross-platform support (fcntl is Unix-only)
"""

import pytest
import asyncio
import json
import tempfile
import signal
import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open
from datetime import datetime

from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import AgentState, JobContext


class TestStatePersistenceBugs:
    """Tests that expose StatePersistence bugs"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def persistence(self, temp_dir):
        """Create StatePersistence instance"""
        state_file = temp_dir / "state.json"
        return StatePersistence(state_file)

    def test_lock_not_released_on_save_exception(self, persistence):
        """
        BUG: If json.dump raises an exception, _release_lock is never called
        because it's only in the finally block after the try-except.

        The current code structure:
        try:
            self._acquire_lock()
            try:
                # write
            except Exception as e:
                raise
            finally:
                self._release_lock()  # This is inside inner try
        except Exception as e:
            # Outer except doesn't release lock

        But actually looking at code again - the finally IS in the right place.
        Let me verify the actual structure...
        """
        sm = AgentStateMachine()
        sm.current_state = AgentState.IDLE

        # Make json.dump fail
        with patch("json.dump", side_effect=IOError("Disk full")):
            with pytest.raises(Exception):  # StatePersistenceError wraps IOError
                persistence.save(sm)

        # Lock should be released even on exception
        # After exception, lock should be None (released by finally block)
        assert persistence._lock_instance is None, (
            "Lock instance not released after save exception - resource leak"
        )

    def test_signal_handler_removed_in_refactoring(self, persistence):
        """
        FIXED: _signal_handler method was removed during refactoring.
        Signal handling is now delegated to the main event loop and asyncio.
        This test documents that the problematic method no longer exists.

        The bug (exit(0) in signal handler) has been fixed by removing
        the signal handler from StatePersistence entirely - it shouldn't
        be the persistence layer's responsibility.
        """
        # Verify the method no longer exists
        assert not hasattr(persistence, "_signal_handler"), (
            "Signal handler should not exist in persistence layer - "
            "should be handled at application level instead"
        )

    def test_no_windows_compatibility(self):
        """
        BUG: StatePersistence uses fcntl which is Unix-only.
        On Windows, this module will fail to import entirely.

        Expected: Should use filelock library for cross-platform support.
        """
        # We can't actually test on Windows here, but we can verify
        # the import dependency exists
        import platform

        if platform.system() == "Windows":
            pytest.skip("Cannot test Windows compatibility on non-Windows system")

        # Verify fcntl is imported (which breaks Windows)
        # This test documents the dependency that causes the issue
        import sensing_voxel.core.persistence as persistence_module

        # Check if fcntl is in the module's namespace
        has_fcntl = hasattr(persistence_module, "fcntl")

        if has_fcntl:
            pytest.fail(
                "BUG: Module uses fcntl which is Unix-only. "
                "This code will fail to import on Windows. "
                "Should use 'filelock' library for cross-platform support."
            )

    def test_load_returns_none_on_json_decode_error(self, temp_dir):
        """
        ISSUE: If the state file is corrupted (invalid JSON), load() returns None
        and logs an error. This silently discards state.

        While returning None is handled by creating fresh state, this could
        mask data corruption issues. Should at least log more prominently
        or raise specific exception.
        """
        state_file = temp_dir / "state.json"

        # Create corrupted JSON file
        state_file.write_text("{invalid json content")

        persistence = StatePersistence(state_file)

        # BUG: Returns None silently - caller has no way to know file was corrupted
        loaded = persistence.load()

        assert loaded is None, "load() should return None for corrupted file (current behavior)"

    def test_load_returns_none_on_missing_required_fields(self, temp_dir):
        """
        FIXED: _from_dict now validates required fields and raises ValueError
        with a clear error message instead of raising KeyError.

        This test documents the fixed behavior: missing 'current_state' raises
        ValueError (caught by load() which logs error and returns None gracefully).
        """
        state_file = temp_dir / "state.json"

        invalid_data = {
            "timestamp": datetime.now().isoformat(),
            # Missing "current_state" field!
            "job_context": None,
        }

        state_file.write_text(json.dumps(invalid_data))
        persistence = StatePersistence(state_file)

        # load() catches ValueError from _from_dict and returns None gracefully
        loaded = persistence.load()

        assert loaded is None, (
            "load() should return None on validation error (ValueError caught and logged)"
        )
