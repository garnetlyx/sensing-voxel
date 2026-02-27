"""
Unit tests for StatePersistence
"""

import pytest
from pathlib import Path
import tempfile
import json
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import AgentState, JobContext, DecisionAction
from datetime import datetime


class TestStatePersistence:
    """Test StatePersistence atomic operations"""

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

    def test_save_and_load_empty_state(self, persistence):
        """Test saving and loading state without job context"""
        sm = AgentStateMachine()
        sm.current_state = AgentState.IDLE

        # Save
        persistence.save(sm)

        # Load
        loaded = persistence.load()

        assert loaded is not None
        assert loaded.current_state == AgentState.IDLE
        assert loaded.job_context is None

    def test_save_and_load_with_job_context(self, persistence):
        """Test saving and loading state with job context"""
        sm = AgentStateMachine()
        sm.current_state = AgentState.AWAITING_APPROVAL
        sm.job_context = JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
        )

        # Save
        persistence.save(sm)

        # Load
        loaded = persistence.load()

        assert loaded is not None
        assert loaded.current_state == AgentState.AWAITING_APPROVAL
        assert loaded.job_context.job_id == "test-123"
        assert loaded.job_context.job_file == "test.stl"

    def test_save_and_load_with_decision_consumed(self, persistence):
        """Test saving and loading consumed decision"""
        sm = AgentStateMachine()
        sm.current_state = AgentState.AWAITING_APPROVAL
        sm.job_context = JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
            decision_consumed=True,
            decision_action=DecisionAction.APPROVE,
        )

        # Save
        persistence.save(sm)

        # Load
        loaded = persistence.load()

        assert loaded is not None
        assert loaded.job_context.decision_consumed

    def test_reset_to_idle_preserves_safety_fields(self, persistence):
        """Test that reset_to_idle preserves safety-critical fields"""
        sm = AgentStateMachine()
        sm.current_state = AgentState.AWAITING_APPROVAL
        sm.job_context = JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
        )
        sm.z_offset_dirty = True
        sm.last_known_z_offset_baseline = 0.15

        # Reset to IDLE
        persistence.reset_to_idle(sm)

        # Reload to verify
        loaded = persistence.load()

        assert loaded.current_state == AgentState.IDLE
        assert loaded.job_context is None  # Job context cleared
        assert loaded.z_offset_dirty  # Safety field preserved
        assert loaded.last_known_z_offset_baseline == 0.15  # Safety field preserved

    def test_atomic_write(self, persistence):
        """Test that writes are atomic (all-or-nothing)"""
        sm1 = AgentStateMachine()
        sm1.current_state = AgentState.IDLE

        # Save first state
        persistence.save(sm1)

        # Verify file exists and is valid JSON
        assert persistence.state_file.exists()
        with open(persistence.state_file, "r") as f:
            data = json.load(f)
        assert data["current_state"] == "IDLE"

        # Save second state
        sm2 = AgentStateMachine()
        sm2.current_state = AgentState.RECEIVING
        persistence.save(sm2)

        # Verify file was replaced, not appended
        with open(persistence.state_file, "r") as f:
            data = json.load(f)
        assert data["current_state"] == "RECEIVING"

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading from non-existent file returns None"""
        state_file = temp_dir / "nonexistent.json"
        persistence = StatePersistence(state_file)

        loaded = persistence.load()

        assert loaded is None

    def test_file_locking(self, persistence):
        """Test that file locking works"""
        # Acquire lock
        persistence._acquire_lock()
        assert persistence._lock_instance is not None
        assert persistence._lock_instance.is_locked

        # Release lock
        persistence._release_lock()
        assert persistence._lock_instance is None

