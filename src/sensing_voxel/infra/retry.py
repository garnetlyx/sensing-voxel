"""
Retry utilities with exponential backoff

Adapted from OpenClaw's src/infra/retry.ts and src/infra/backoff.ts patterns
"""

import asyncio
import random
from typing import (
    Callable,
    TypeVar,
    Awaitable,
    Optional,
    Tuple,
    Type,
)
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class BackoffPolicy:
    """Backoff policy configuration"""
    base_delay: float = 2.0      # Initial delay in seconds
    max_delay: float = 30.0      # Maximum delay in seconds
    factor: float = 1.8          # Multiplicative factor
    jitter: float = 0.25         # Jitter factor (0.0 to 1.0)


def compute_backoff(
    attempt: int,
    policy: Optional[BackoffPolicy] = None,
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

    # Exponential backoff
    delay = policy.base_delay * (policy.factor ** attempt)
    delay = min(delay, policy.max_delay)

    # Add jitter
    if policy.jitter > 0:
        jitter_amount = delay * policy.jitter * random.random()
        delay += jitter_amount

    return delay


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    policy: Optional[BackoffPolicy] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """
    Retry an async function with exponential backoff

    Args:
        fn: Async function to retry
        max_attempts: Maximum number of attempts (including first)
        policy: Backoff policy configuration
        retryable_exceptions: Exception types to retry on. If None, retries on all exceptions
        on_retry: Callback invoked before each retry with attempt number and exception

    Returns:
        Result of the function

    Raises:
        Exception: Last exception after all retries exhausted
    """
    if policy is None:
        policy = BackoffPolicy()

    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    last_exception: Optional[Exception] = None

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


async def retry_async_with_reset(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    policy: Optional[BackoffPolicy] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_success_reset: Optional[Callable[[], None]] = None,
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
