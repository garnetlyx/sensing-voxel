"""
Unit tests for DecisionEngine
"""

import asyncio
import pytest
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    DecisionAction,
    ConsumeResult,
    AgentState,
    JobContext,
)
from datetime import datetime


class TestDecisionEngine:
    """Test DecisionEngine idempotency and timeouts"""

    @pytest.fixture
    def engine(self):
        """Create DecisionEngine instance"""
        return DecisionEngine()

    @pytest.fixture
    def job_context(self):
        """Create test JobContext"""
        return JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
        )

    @pytest.mark.asyncio
    async def test_consume_decision_success(self, engine, job_context):
        """Test successful decision consumption"""
        result = await engine.consume_decision(
            job_context=job_context,
            decision_token="test-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        assert result == ConsumeResult.CONSUMED
        assert job_context.decision_consumed
        assert job_context.decision_action == DecisionAction.APPROVE

    @pytest.mark.asyncio
    async def test_consume_decision_already_consumed(self, engine, job_context):
        """Test that already-consumed decisions are rejected"""
        # First consumption
        result1 = await engine.consume_decision(
            job_context=job_context,
            decision_token="test-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        assert result1 == ConsumeResult.CONSUMED

        # Second consumption should fail
        result2 = await engine.consume_decision(
            job_context=job_context,
            decision_token="test-token",
            action=DecisionAction.REJECT,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        assert result2 == ConsumeResult.ALREADY_HANDLED

    @pytest.mark.asyncio
    async def test_consume_decision_invalid_token(self, engine, job_context):
        """Test that invalid tokens are rejected"""
        result = await engine.consume_decision(
            job_context=job_context,
            decision_token="wrong-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        assert result == ConsumeResult.INVALID_TOKEN
        assert not job_context.decision_consumed

    @pytest.mark.asyncio
    async def test_consume_decision_state_mismatch(self, engine, job_context):
        """Test that state mismatches are rejected"""
        result = await engine.consume_decision(
            job_context=job_context,
            decision_token="test-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.PRINTING,  # Wrong state
        )

        assert result == ConsumeResult.INVALID_STATE
        assert not job_context.decision_consumed

    @pytest.mark.asyncio
    async def test_consume_decision_timeout_action(self, engine, job_context):
        """Test that timeout actions skip state validation"""
        result = await engine.consume_decision(
            job_context=job_context,
            decision_token="test-token",
            action=DecisionAction.TIMEOUT_APPROVAL,
            expected_state=None,  # Timeout actions don't require state check
            current_state=None,
        )

        assert result == ConsumeResult.CONSUMED
        assert job_context.decision_consumed
        assert job_context.decision_action == DecisionAction.TIMEOUT_APPROVAL

    @pytest.mark.asyncio
    async def test_concurrent_consumption(self, engine, job_context):
        """Test that concurrent consumption attempts are serialized"""
        results = []

        async def consume_attempt(action):
            result = await engine.consume_decision(
                job_context=job_context,
                decision_token="test-token",
                action=action,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )
            results.append(result)

        # Start two concurrent attempts
        await asyncio.gather(
            consume_attempt(DecisionAction.APPROVE),
            consume_attempt(DecisionAction.REJECT),
        )

        # Exactly one should succeed
        assert results.count(ConsumeResult.CONSUMED) == 1
        assert results.count(ConsumeResult.ALREADY_HANDLED) == 1

    @pytest.mark.asyncio
    async def test_schedule_timeout(self, engine, job_context):
        """Test timeout scheduling"""
        engine.schedule_timeout(
            job_id="test-123",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        assert "test-123" in engine._timeout_tasks
        assert len(engine._timeout_tasks) == 1

    @pytest.mark.asyncio
    async def test_cancel_timeout(self, engine, job_context):
        """Test timeout cancellation"""
        engine.schedule_timeout(
            job_id="test-123",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        assert "test-123" in engine._timeout_tasks

        engine.cancel_timeout("test-123")

        assert "test-123" not in engine._timeout_tasks

    def test_set_timeouts(self, engine):
        """Test timeout configuration"""
        engine.set_timeouts(
            approval_timeout_seconds=150,
        )
        (approval_timeout,) = engine.get_timeouts()
        assert approval_timeout == 150

    @pytest.mark.asyncio
    async def test_timeout_preemption_by_user(self, engine, job_context):
        """Test that user decision cancels pending timeout"""
        # Schedule timeout
        engine.schedule_timeout(
            job_id="test-123",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        assert "test-123" in engine._timeout_tasks

        # User makes decision before timeout
        result = await engine.consume_decision(
            job_context=job_context,
            decision_token="test-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        assert result == ConsumeResult.CONSUMED
        assert "test-123" not in engine._timeout_tasks  # Timeout cancelled

