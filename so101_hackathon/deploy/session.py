"""Generic teleop deploy loop."""

from __future__ import annotations

import time

from so101_hackathon.deploy.runtime import (
    DEFAULT_FPS,
    FixedDisturbanceChannel,
    LiveTeleopObservationBuilder,
    blend_with_leader,
    build_follower_action,
    clamp_joint_positions,
    hardware_obs_to_joint_positions,
    normalize_controller_action,
)
from so101_hackathon.utils.rl_utils import clamp_action


def run_deploy_session(
    *,
    args,
    leader,
    follower,
    controller,
    observation_builder: LiveTeleopObservationBuilder,
    metrics,
    lower_limits,
    upper_limits,
    sleep_fn,
    active_follower_joint_names=None,
    time_fn=time.perf_counter,
    num_iterations: int | None = None,
) -> int:
    start_time = time_fn()
    previous_sample_time = None
    observation_builder.reset()
    controller.reset()
    disturbance_channel = FixedDisturbanceChannel(
        delay_steps=int(getattr(args, "delay_steps", 0)),
        noise_std=float(getattr(args, "noise_std", 0.0)),
        seed=int(getattr(args, "seed", 0)),
    )
    disturbance_channel.reset()
    iter_idx = 0

    while True:
        if num_iterations is not None and iter_idx >= num_iterations:
            break

        loop_start = time_fn()
        follower_observation = follower.get_observation()
        leader_observation = leader.get_action()
        sample_time = time_fn()
        dt = (
            1.0 / max(int(getattr(args, "fps", DEFAULT_FPS)), 1)
            if previous_sample_time is None
            else max(sample_time - previous_sample_time, 1.0e-6)
        )
        live_obs = observation_builder.build(
            leader_observation=leader_observation,
            follower_observation=follower_observation,
            dt=dt,
        )
        controller_action = normalize_controller_action(
            controller.act(live_obs.observation))
        if getattr(controller, "action_mode", "absolute") == "residual":
            residual_action = clamp_action(
                controller_action,
                limit=1.0,
            )
            if hasattr(residual_action, "tolist"):
                residual_action = residual_action.tolist()
            blended_action = [
                float(leader) + float(args.controller_coeff) * float(residual)
                for leader, residual in zip(live_obs.leader_joint_pos, residual_action)
            ]
        else:
            blended_action = blend_with_leader(
                live_obs.leader_joint_pos,
                controller_action,
                float(args.controller_coeff),
            )
        disturbed_action = disturbance_channel.apply(blended_action)
        commanded_joint_pos = clamp_joint_positions(
            disturbed_action, lower_limits, upper_limits)
        follower_action = build_follower_action(
            commanded_joint_pos,
            active_joint_names=active_follower_joint_names,
        )
        follower.send_action(follower_action)
        observation_builder.set_previous_action(commanded_joint_pos)

        metrics.update(
            step=iter_idx,
            timestamp_s=sample_time - start_time,
            leader_joint_pos=live_obs.leader_joint_pos,
            follower_joint_pos=live_obs.follower_joint_pos,
            commanded_joint_pos=commanded_joint_pos,
        )
        iter_idx += 1

        if getattr(args, "print_every", 0) > 0 and iter_idx % int(args.print_every) == 0:
            loop_dt = max(time_fn() - loop_start, 1.0e-8)
            hz = 1.0 / loop_dt
            print(metrics.format_status_line(iter_idx=iter_idx, hz=hz))
            print("  " + metrics.format_last_joint_errors())

        elapsed = time_fn() - loop_start
        sleep_fn(
            max(1.0 / max(int(getattr(args, "fps", DEFAULT_FPS)), 1) - elapsed, 0.0))
        previous_sample_time = sample_time

        if getattr(args, "teleop_time_s", None) is not None and sample_time - start_time >= float(args.teleop_time_s):
            break

    return iter_idx
