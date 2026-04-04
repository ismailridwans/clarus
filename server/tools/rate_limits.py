"""Rate limit state management for Clarus environment.

Tools can be rate-limited at the start of an episode (Task 3) or after
a burst of calls.  The rate state maps action_type → first_available_step.
A tool is unavailable when step_number <= first_available_step.
"""

from __future__ import annotations

from typing import Dict, List


RateState = Dict[str, int]  # action_type → first_available_step


def init_rate_state(
    rate_limited_at_start: List[str],
    cooldown_steps: int,
    start_step: int = 1,
) -> RateState:
    """Build initial rate state from the episode configuration.

    Tool available when step_number > first_available_step.
    So: first_available_step = start_step + cooldown_steps means the tool
    is blocked on steps 1..cooldown_steps and available from step
    start_step + cooldown_steps + 1 onward.

    Args:
        rate_limited_at_start: Action types to block at episode start.
        cooldown_steps: Number of steps each listed tool is unavailable.
        start_step: The step number at episode start (usually 1).

    Returns:
        Dict mapping each rate-limited action type to its first_available_step.
    """
    return {
        action: start_step + cooldown_steps
        for action in rate_limited_at_start
    }


def is_rate_limited(
    action_type: str, step_number: int, rate_state: RateState
) -> bool:
    """Return True if action_type is blocked at the given step.

    Args:
        action_type: The action being attempted.
        step_number: Current step number (1-indexed).
        rate_state: Current rate state dict.

    Returns:
        True if the action is currently rate-limited.
    """
    first_available = rate_state.get(action_type)
    if first_available is None:
        return False
    return step_number <= first_available


def get_cooldown_remaining(
    action_type: str, step_number: int, rate_state: RateState
) -> int:
    """Return how many more steps until action_type becomes available.

    Returns 0 if the action is not rate-limited.

    Args:
        action_type: The action type to check.
        step_number: Current step number.
        rate_state: Current rate state dict.

    Returns:
        Number of steps remaining in cooldown (0 if available).
    """
    first_available = rate_state.get(action_type)
    if first_available is None or step_number > first_available:
        return 0
    return first_available - step_number + 1


def get_cooldown_status(
    rate_state: RateState, step_number: int
) -> Dict[str, int]:
    """Return a dict of action_type → cooldown_steps_remaining for all blocked tools.

    Only includes tools that are currently rate-limited (remaining > 0).

    Args:
        rate_state: Current rate state dict.
        step_number: Current step number.

    Returns:
        Dict of currently-blocked tools with their remaining cooldown steps.
    """
    return {
        action: get_cooldown_remaining(action, step_number, rate_state)
        for action in rate_state
        if is_rate_limited(action, step_number, rate_state)
    }


def get_rate_limited_tools(
    rate_state: RateState, step_number: int
) -> List[str]:
    """Return list of action types currently blocked by rate limiting.

    Args:
        rate_state: Current rate state dict.
        step_number: Current step number.

    Returns:
        List of currently rate-limited action type names.
    """
    return [
        action
        for action in rate_state
        if is_rate_limited(action, step_number, rate_state)
    ]
