"""
Integration tests for decision race conditions.

Tests concurrent timeout + user action, double-approve, etc.
"""

import pytest
import asyncio
from datetime import datetime

from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    AgentState,
    ConsumeResult,
    DecisionAction,
    JobContext,
    generate_decision_token,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine


class TestDecisionRaces:
    """Test decision race conditions."""

    @pytest.fixture
    def setup(self, tmp_path):
        sm = AgentStateMachine()
        persistence = StatePersistence(tmp_path / "state.json")
        engine = DecisionEngine()
        return sm, persistence, engine

    @pytest.fixture
    def awaiting_job(self, setup):
        sm, persistence, engine = setup
        job = JobContext(
            job_id=generate_job_id(),
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
        )
        sm.job_context = job
        sm.current_state = AgentState.AWAITING_APPROVAL
        return sm, persistence, engine, job

    @pytest.mark.asyncio
    async def test_double_approve(self, awaiting_job):
        """Second approve returns ALREADY_HANDLED."""
        sm, _, engine, job = awaiting_job

        # First approve
        r1 = await engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert r1 == ConsumeResult.CONSUMED

        # Second approve
        r2 = await engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert r2 == ConsumeResult.ALREADY_HANDLED

    @pytest.mark.asyncio
    async def test_approve_then_reject(self, awaiting_job):
        """Reject after approve returns ALREADY_HANDLED."""
        sm, _, engine, job = awaiting_job

        r1 = await engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert r1 == ConsumeResult.CONSUMED

        r2 = await engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.REJECT,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert r2 == ConsumeResult.ALREADY_HANDLED

    @pytest.mark.asyncio
    async def test_concurrent_approve(self, awaiting_job):
        """Only one of two concurrent approves succeeds."""
        sm, _, engine, job = awaiting_job

        async def approve():
            return await engine.consume_decision(
                job_context=job,
                decision_token=job.decision_token,
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=sm.current_state,
            )

        results = await asyncio.gather(approve(), approve())

        consumed = sum(1 for r in results if r == ConsumeResult.CONSUMED)
        already = sum(1 for r in results if r == ConsumeResult.ALREADY_HANDLED)

        assert consumed == 1
        assert already == 1

    @pytest.mark.asyncio
    async def test_wrong_token_rejected(self, awaiting_job):
        """Wrong decision token is rejected."""
        sm, _, engine, job = awaiting_job

        result = await engine.consume_decision(
            job_context=job,
            decision_token="wrong-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert result == ConsumeResult.INVALID_TOKEN

    @pytest.mark.asyncio
    async def test_state_mismatch_rejected(self, awaiting_job):
        """Action with wrong state expectation is rejected."""
        sm, _, engine, job = awaiting_job
        sm.current_state = AgentState.IDLE  # Change state

        result = await engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert result == ConsumeResult.INVALID_STATE
