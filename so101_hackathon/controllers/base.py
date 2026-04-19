"""Shared controller interface.

Students only need to understand this contract:

1. ``reset()`` is called at the start of an episode.
2. ``act(obs)`` receives the environment observation tensor.
3. ``act(obs)`` must return an action tensor with the same semantics used by the
   training environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseController(ABC):
    """Minimal controller interface for the hackathon repo."""

    def reset(self) -> None:
        """Reset any controller-side episode state.

        Most beginner controllers can leave this method unchanged.
        """

    @abstractmethod
    def act(self, obs: Any) -> Any:
        """Return one action for the provided observation."""

