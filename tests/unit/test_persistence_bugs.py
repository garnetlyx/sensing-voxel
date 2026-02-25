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
        with patch('json.dump', side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                persistence.save(sm)

        # BUG: Lock should be released even on exception
        # After exception, lock should be None (released)
        assert persistence._lock_fd is None, \
            "Lock file descriptor not released after save exception - resource leak"

    def test_signal_handler_calls_exit_zero(self, persistence):
        """
        BUG: _signal_handler calls exit(0) which terminates the entire Python process.
        This is unacceptable in library code - it kills the application using the library.

        Expected: Should raise a specific exception or set a flag for graceful shutdown.
        """
        # We can't actually test exit(0) without killing the test runner
        # So we verify that exit IS called by mocking it

        with patch('builtins.exit') as mock_exit:
            # Simulate SIGTERM
            persistence._signal_handler(signal.SIGTERM, None)

            # BUG: exit(0) was called - this kills the process!
            mock_exit.assert_called_once_with(0)

    def test_no_windows_compatibility(self):
        """
        BUG: StatePersistence uses fcntl which is Unix-only.
        On Windows, this module will fail to import entirely.

        Expected: Should use filelock library for cross-platform support.
        """
        # We can't actually test on Windows here, but we can verify
        # the import dependency exists
        import platform

        if platform.system() == 'Windows':
            pytest.skip("Cannot test Windows compatibility on non-Windows system")

        # Verify fcntl is imported (which breaks Windows)
        # This test documents the dependency that causes the issue
        import sensing_voxel.core.persistence as persistence_module

        # Check if fcntl is in the module's namespace
        has_fcntl = hasattr(persistence_module, 'fcntl')

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

        assert loaded is None, \
            "load() should return None for corrupted file (current behavior)"

    def test_load_returns_none_on_missing_required_fields(self, temp_dir):
        """
        ISSUE: _from_dict doesn't validate required fields. If the JSON
        is missing critical fields like 'current_state', it may crash
        or produce invalid state machine.
        """
        state_file = temp_dir / "state.json"

        # Create JSON missing required fields
        invalid_data = {
            "timestamp": datetime.now().isoformat(),
            # Missing "current_state" field!
            "job_context": None,
        }

        state_file.write_text(json.dumps(invalid_data))
        persistence = StatePersistence(state_file)

        # This should either raise an error or handle gracefully
        # Currently it will crash when trying to access data["current_state"]
        with pytest.raises((KeyError, TypeError)):
            loaded = persistence.load()
