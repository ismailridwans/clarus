"""Clarus OpenEnv client — HTTP wrapper for the REST API.

Provides a simple programmatic interface to the Clarus environment
running at any base URL (local or HuggingFace Space).

Example::

    from client import ClarusClient

    env = ClarusClient("https://ismailridwans-clarus.hf.space")
    obs = env.reset(task_name="deductive_liability", seed=1001)
    print(obs["patient_complaint"])

    result = env.step({"action_type": "authenticate_patient",
                       "parameters": {"patient_id": obs["case_id"]}})
    print(result["observation"], result["reward"])
    env.close()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class ClarusClient:
    """HTTP client for the Clarus OpenEnv REST API.

    Args:
        base_url: Base URL of a running Clarus server, e.g.
                  ``"http://localhost:7860"`` or
                  ``"https://ismailridwans-clarus.hf.space"``.
        timeout:  HTTP request timeout in seconds (default 30).
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a new episode.

        Args:
            task_name: One of ``"deductive_liability"``,
                       ``"abductive_conflict"``,
                       ``"adversarial_fabrication"``.
                       Defaults to ``"deductive_liability"``.
            seed:      Deterministic seed.  Defaults to a random train seed.

        Returns:
            The initial observation dict (same structure as
            ``result["observation"]`` from :meth:`step`).
        """
        body: Dict[str, Any] = {}
        if task_name is not None:
            body["task_name"] = task_name
        if seed is not None:
            body["seed"] = seed

        r = self._client.post(f"{self.base_url}/reset", json=body)
        r.raise_for_status()
        data = r.json()
        # Flatten for convenience: merge top-level metadata into observation
        obs = data.get("observation", {})
        obs["episode_id"] = data.get("episode_id")
        obs["task_name"] = data.get("task_name")
        obs["seed"] = data.get("seed")
        return obs

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one action.

        Args:
            action: Dict with ``action_type`` (str) and ``parameters``
                    (dict).  Example::

                        {
                            "action_type": "fetch_eob",
                            "parameters": {}
                        }

        Returns:
            Dict with keys ``observation``, ``reward``, ``done``, ``info``.
        """
        r = self._client.post(f"{self.base_url}/step", json={"action": action})
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """Return the current environment state (episode metadata).

        Returns:
            Dict with ``episode_id``, ``task_name``, ``seed``,
            ``step_number``, ``done``, ``observation``.
        """
        r = self._client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, str]:
        """Check that the server is healthy.

        Returns:
            ``{"status": "healthy", "service": "clarus"}``
        """
        r = self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    # Context-manager support
    def __enter__(self) -> "ClarusClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
