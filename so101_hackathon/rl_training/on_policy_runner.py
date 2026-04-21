"""Optional local runner wrapper.

The hackathon repo keeps this file as a small extension point, but unlike the
source project it does not require extra W&B metadata plumbing to function.
"""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner


class OnPolicyRunner(RslRlOnPolicyRunner):  # type: ignore
    """Thin wrapper kept for future repo-local hooks."""

    def learn(self, *args, **kwargs):
        return super().learn(*args, **kwargs)
