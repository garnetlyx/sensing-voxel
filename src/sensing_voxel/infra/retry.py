"""
Retry utilities with exponential backoff

Adapted from OpenClaw's src/infra/retry.ts and src/infra/backoff.ts patterns
"""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class BackoffPolicy:
    """Backoff policy configuration"""

    base_delay: float = 2.0  # Initial delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    factor: float = 1.8  # Multiplicative factor
    jitter: float = 0.25  # Jitter factor (0.0 to 1.0)

    def __post_init__(self):
        """Validate policy parameters"""
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.factor <= 1.0:
            raise ValueError("factor must be > 1.0")
        if not (0.0 <= self.jitter <= 1.0):
            raise ValueError("jitter must be in range [0.0, 1.0]")


def compute_backoff(
    attempt: int,
    policy: BackoffPolicy | None = None,
) -> float:
    """
    Compute backoff delay with exponential backoff and jitter

    Formula:
        delay = min(max_delay, base_delay * factor^attempt)
        jitter = delay * jitter * random()
        final = delay + jitter

    Args:
        attempt: Current retry attempt number (0-indexed)
        policy: Backoff policy configuration

    Returns:
        Delay in seconds to wait before next attempt
    """
    if policy is None:
        policy = BackoffPolicy()

    # Cap attempt to prevent overflow in exponentiation
    # After log(max_delay / base_delay) / log(factor) attempts,
    # we've already reached max_delay, so further attempts don't matter
    max_useful_attempts = (
        int(math.log(policy.max_delay / policy.base_delay) / math.log(policy.factor)) + 1
    )
    capped_attempt = min(attempt, max_useful_attempts)

    # Exponential backoff
    delay = policy.base_delay * (policy.factor**capped_attempt)
    delay = min(delay, policy.max_delay)

    # Add jitter
    if policy.jitter > 0:
        jitter_amount = delay * policy.jitter * random.random()
        delay += jitter_amount

    return delay


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    policy: BackoffPolicy | None = None,
    backoff_policy: BackoffPolicy | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """
    Retry an async function with exponential backoff

    Args:
        fn: Async function to retry
        max_attempts: Maximum number of attempts (including first)
        policy: Backoff policy configuration
        backoff_policy: Backward-compatible alias for policy
        retryable_exceptions: Exception types to retry on. If None, retries on all exceptions
        on_retry: Callback invoked before each retry with attempt number and exception

    Returns:
        Result of the function

    Raises:
        Exception: Last exception after all retries exhausted
    """
    if max_attempts < 1:
        raise TypeError("max_attempts must be at least 1")

    if policy is not None and backoff_policy is not None:
        raise TypeError("Specify only one of policy or backoff_policy")

    if policy is None:
        policy = backoff_policy or BackoffPolicy()

    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable_exceptions as e:
            last_exception = e

            # No more retries
            if attempt >= max_attempts - 1:
                break

            # Invoke retry callback
            if on_retry:
                on_retry(attempt, e)

            # Compute and wait for backoff
            delay = compute_backoff(attempt, policy)
            await asyncio.sleep(delay)

    # All retries exhausted
    raise last_exception  # type: ignore


async def retry_async_with_reset[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    policy: BackoffPolicy | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    on_success_reset: Callable[[], None] | None = None,
) -> T:
    """
    Retry with automatic attempt counter reset on success

    This variant is useful for reconnection loops where you want to
    reset the backoff counter after a successful operation.

    Args:
        fn: Async function to retry
        max_attempts: Maximum number of attempts
        policy: Backoff policy
        retryable_exceptions: Exception types to retry on
        on_retry: Callback before each retry
        on_success_reset: Callback to reset state on success (e.g., set reconnect_attempts = 0)

    Returns:
        Result of the function

    Raises:
        Exception: Last exception after all retries exhausted
    """
    result = await retry_async(
        fn,
        max_attempts=max_attempts,
        policy=policy,
        retryable_exceptions=retryable_exceptions,
        on_retry=on_retry,
    )

    if on_success_reset:
        on_success_reset()

    return result
