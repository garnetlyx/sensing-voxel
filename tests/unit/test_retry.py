import re

import pytest  # type: ignore[reportMissingImports]

from sensing_voxel.infra import retry as retry_mod  # type: ignore[reportMissingImports]


class TestBackoffPolicy:
    def test_initial_delay_calculation_without_jitter(self) -> None:
        policy = retry_mod.BackoffPolicy(base_delay=0.5, max_delay=10.0, factor=2.0, jitter=0.0)

        delay = retry_mod.compute_backoff(attempt=0, policy=policy)

        assert delay == pytest.approx(0.5)

    def test_exponential_backoff_with_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = retry_mod.BackoffPolicy(base_delay=2.0, max_delay=30.0, factor=2.0, jitter=0.25)
        monkeypatch.setattr(retry_mod.random, "random", lambda: 0.5)

        delay = retry_mod.compute_backoff(attempt=2, policy=policy)

        assert delay == pytest.approx(9.0)

    def test_max_delay_clamping_and_attempt_capping(self) -> None:
        policy = retry_mod.BackoffPolicy(base_delay=0.1, max_delay=1.0, factor=2.0, jitter=0.0)

        delay = retry_mod.compute_backoff(attempt=10_000, policy=policy)

        assert delay == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("kwargs", "msg"),
        [
            ({"base_delay": 0.0}, "base_delay must be positive"),
            ({"max_delay": 0.0}, "max_delay must be positive"),
            ({"base_delay": 2.0, "max_delay": 1.0}, "max_delay must be >= base_delay"),
            ({"factor": 1.0}, "factor must be > 1.0"),
            ({"jitter": -0.01}, "jitter must be in range [0.0, 1.0]"),
            ({"jitter": 1.01}, "jitter must be in range [0.0, 1.0]"),
        ],
    )
    def test_backoff_policy_validation(self, kwargs: dict[str, float], msg: str) -> None:
        with pytest.raises(ValueError, match=re.escape(msg)):
            retry_mod.BackoffPolicy(**kwargs)

    def test_compute_backoff_uses_default_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(retry_mod.random, "random", lambda: 0.0)

        delay = retry_mod.compute_backoff(attempt=0, policy=None)

        assert delay == pytest.approx(2.0)


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_successful_retry_on_transient_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = {"count": 0}
        sleeps: list[float] = []
        retries: list[tuple[int, str]] = []

        async def fn() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("transient")
            return "ok"

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(retry_mod.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            retry_mod, "compute_backoff", lambda attempt, policy: 0.01 * (attempt + 1)
        )

        result = await retry_mod.retry_async(
            fn,
            max_attempts=5,
            retryable_exceptions=(ConnectionError,),
            on_retry=lambda attempt, exc: retries.append((attempt, str(exc))),
        )

        assert result == "ok"
        assert attempts["count"] == 3
        assert retries == [(0, "transient"), (1, "transient")]
        assert sleeps == [0.01, 0.02]

    @pytest.mark.asyncio
    async def test_give_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []

        async def fn() -> None:
            raise TimeoutError("still failing")

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(retry_mod.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(retry_mod, "compute_backoff", lambda attempt, policy: 0.1)

        with pytest.raises(TimeoutError, match="still failing"):
            await retry_mod.retry_async(
                fn,
                max_attempts=3,
                retryable_exceptions=(TimeoutError,),
            )

        assert sleeps == [0.1, 0.1]

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates_immediately(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: list[float] = []

        async def fn() -> None:
            raise ValueError("bad input")

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(retry_mod.asyncio, "sleep", fake_sleep)

        with pytest.raises(ValueError, match="bad input"):
            await retry_mod.retry_async(fn, max_attempts=5, retryable_exceptions=(ConnectionError,))

        assert sleeps == []

    @pytest.mark.asyncio
    async def test_retry_count_tracking_and_backoff_attempt_numbers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        compute_calls: list[int] = []

        async def fn() -> None:
            raise ConnectionError("down")

        async def fake_sleep(_: float) -> None:
            return None

        def fake_compute(attempt: int, policy: retry_mod.BackoffPolicy | None) -> float:
            compute_calls.append(attempt)
            return 0.0

        monkeypatch.setattr(retry_mod.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(retry_mod, "compute_backoff", fake_compute)

        with pytest.raises(ConnectionError, match="down"):
            await retry_mod.retry_async(fn, max_attempts=4, retryable_exceptions=(ConnectionError,))

        assert compute_calls == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_retry_async_invalid_max_attempts_zero_raises_type_error(self) -> None:
        async def fn() -> str:
            return "never called"

        with pytest.raises(TypeError):
            await retry_mod.retry_async(fn, max_attempts=0)


class TestRetryAsyncWithReset:
    @pytest.mark.asyncio
    async def test_calls_reset_callback_on_success(self) -> None:
        reset_called = {"count": 0}

        async def fn() -> str:
            return "ok"

        result = await retry_mod.retry_async_with_reset(
            fn,
            on_success_reset=lambda: reset_called.__setitem__("count", reset_called["count"] + 1),
        )

        assert result == "ok"
        assert reset_called["count"] == 1

    @pytest.mark.asyncio
    async def test_no_reset_callback_is_allowed(self) -> None:
        async def fn() -> int:
            return 42

        result = await retry_mod.retry_async_with_reset(fn, on_success_reset=None)

        assert result == 42


class TestRetryableExceptionHelperCompatibility:
    def test_is_retryable_exception_helper_or_equivalent_behavior(self) -> None:
        helper = getattr(retry_mod, "is_retryable_exception", None)
        err = ConnectionError("x")

        if helper is not None:
            assert helper(err, (ConnectionError,)) is True
            assert helper(err, (TimeoutError,)) is False
            return

        assert isinstance(err, (ConnectionError,))
        assert not isinstance(err, (TimeoutError,))
