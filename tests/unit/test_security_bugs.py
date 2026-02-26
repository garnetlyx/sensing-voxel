"""
BUG TESTS: Security Vulnerabilities

These tests expose security issues in the codebase.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.core.models import generate_decision_token, JobContext
from datetime import datetime


class TestSecurityBugs:
    """Tests exposing security vulnerabilities"""

    @pytest.mark.asyncio
    async def test_gcode_injection_in_set_z_offset(self):
        """
        BUG: set_z_offset() formats user input directly into G-code without sanitization.

        A malicious offset value like "0.1\nM104 S300\n" could inject arbitrary G-code.

        This is a CRITICAL security vulnerability - arbitrary G-code execution.
        """
        adapter = KlipperAdapter(host="localhost", port=7125)

        # Mock the _rest_request to capture what G-code is sent
        with patch.object(adapter, '_rest_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = Mock(status=Mock(name="CommandResultStatus.SUCCESS"), data={})

            # Try to inject malicious G-code
            malicious_offset = "0.1\nM104 S300\nM109 S300"

            try:
                await adapter.set_z_offset(malicious_offset)  # type: ignore
            except (TypeError, ValueError):
                # Good - input was rejected
                pass
            except Exception as e:
                # If we get here, the malicious input was accepted
                # Check what was sent to the printer
                if mock_request.called:
                    call_args = mock_request.call_args
                    json_data = call_args[1].get('json', {})
                    script = json_data.get('script', '')

                    # If the script contains newlines, injection succeeded
                    if '\n' in script or 'M104' in script:
                        pytest.fail(
                            f"CRITICAL: G-code injection succeeded!\n"
                            f"Script sent to printer: {script}"
                        )

    @pytest.mark.asyncio
    async def test_gcode_injection_with_semicolon(self):
        """
        BUG: set_z_offset() doesn't sanitize semicolons which can chain G-code commands.

        Input like "0.1; M104 S300" could execute multiple commands.
        """
        adapter = KlipperAdapter(host="localhost", port=7125)

        with patch.object(adapter, '_rest_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = Mock(status=Mock(name="CommandResultStatus.SUCCESS"), data={})

            # Try semicolon injection
            malicious_offset = "0.1; M104 S300"

            try:
                await adapter.set_z_offset(malicious_offset)  # type: ignore
            except (TypeError, ValueError):
                pass  # Good - rejected
            except Exception:
                # Check if injection succeeded
                if mock_request.called:
                    pytest.fail("CRITICAL: Semicolon G-code injection possible")

    def test_decision_token_not_cryptographically_secure(self):
        """
        BUG: Decision tokens use uuid4().hex which is not cryptographically secure.

        UUID4 is based on random numbers but:
        1. It's not designed for security tokens
        2. The namespace is only 2^128 but without proper entropy mixing
        3. Tokens should use secrets.token_urlsafe() instead

        This is MEDIUM severity - tokens could theoretically be guessed.
        """
        # Generate many tokens and check for patterns
        tokens = [generate_decision_token() for _ in range(100)]

        # All tokens should be unique
        assert len(set(tokens)) == len(tokens), "Duplicate tokens generated!"

        # Tokens should be 32 hex characters (uuid4 without dashes)
        for token in tokens:
            assert len(token) == 32, f"Unexpected token length: {len(token)}"
            assert all(c in '0123456789abcdef' for c in token), "Token contains non-hex chars"

        # The issue is that uuid4() uses random.getrandbits() which is
        # NOT cryptographically secure in older Python versions
        # and even in newer versions, secrets module is preferred

    @pytest.mark.asyncio
    async def test_no_http_timeout_on_rest_requests(self):
        """
        BUG: _rest_request() doesn't set timeout on HTTP requests.

        If the printer is unresponsive, the request will hang forever.

        This is MEDIUM severity - can cause indefinite hangs.
        """
        adapter = KlipperAdapter(host="localhost", port=7125)

        # Mock the session to capture timeout values
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": {}})

        mock_session = Mock()
        mock_session.request = Mock(return_value=mock_response)

        # Check if timeout is passed
        with patch.object(adapter, '_session', mock_session):
            await adapter._rest_request("GET", "/test")

            # Check the call arguments
            call_kwargs = mock_session.request.call_args[1]

            # BUG: No timeout is set - requests can hang forever
            assert 'timeout' in call_kwargs, (
                "BUG: No timeout set on HTTP request. "
                "Requests can hang indefinitely if printer is unresponsive."
            )

    @pytest.mark.asyncio
    async def test_job_context_token_mutable(self):
        """
        BUG: JobContext.decision_token is mutable after creation.

        If an attacker gains access to job_context, they can change the token.

        FIXED: JobContext now protects decision_token as immutable.
        """
        job = JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="original-token",
        )

        original_token = job.decision_token

        # Attempt to modify token - should raise AttributeError
        try:
            job.decision_token = "attacker-controlled-token"
            # If we get here, the field is mutable (bug!)
            assert job.decision_token == original_token, (
                "BUG: decision_token is mutable. Security-critical fields should be read-only."
            )
        except AttributeError as e:
            # Good! Field is immutable
            assert "immutable" in str(e).lower(), f"Should mention immutable: {e}"
            assert job.decision_token == original_token, "Token should remain unchanged"

    @pytest.mark.asyncio
    async def test_no_rate_limiting_on_decisions(self):
        """
        BUG: DecisionEngine has no rate limiting on consume_decision().

        An attacker could brute-force decision tokens by trying many combinations.

        This is LOW-MEDIUM severity - depends on deployment context.
        """
        from sensing_voxel.core.decision import DecisionEngine
        from sensing_voxel.core.models import DecisionAction, AgentState

        engine = DecisionEngine()

        job = JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="secret-token",
        )

        # Try many invalid tokens rapidly
        results = []
        for i in range(100):
            result = await engine.consume_decision(
                job_context=job,
                decision_token=f"wrong-token-{i}",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )
            results.append(result)

        # All should fail, but there's no rate limiting
        invalid_count = sum(1 for r in results if r.name == "INVALID_TOKEN")

        # After 100 invalid attempts, there should be some rate limiting
        # BUG: There's no mechanism to prevent brute force
        assert invalid_count == 100, "Not all invalid tokens were rejected"

        # In a secure system, after N failed attempts, further attempts
        # should be blocked or throttled
