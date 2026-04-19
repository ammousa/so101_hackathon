"""Wrapper adapter to keep nsrl-like wrapper invocation style."""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as IsaacLabRslRlVecEnvWrapper


def RslRlVecEnvWrapper(env, args_cli=None, **kwargs):
    """Construct upstream Isaac Lab RSL-RL wrapper.

    Supports nsrl-style invocation: ``wrapper(env, args_cli=args_cli)``.
    """
    clip_actions = kwargs.pop("clip_actions", None)
    if clip_actions is None and args_cli is not None:
        clip_actions = getattr(args_cli, "clip_actions", None)

    if clip_actions is None:
        return IsaacLabRslRlVecEnvWrapper(env)
    return IsaacLabRslRlVecEnvWrapper(env, clip_actions=clip_actions)
