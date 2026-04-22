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
from so101_hackathon.deploy.ultrazohm import UltraZohmDisturbanceChannel
from so101_hackathon.utils.rl_utils import TELEOP_RESIDUAL_ACTION_SCALE, clamp_action


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
    """Run run deploy session."""
    start_time = time_fn()
    previous_sample_time = None
    observation_builder.reset()
    controller.reset()
    disturbance_channel_name = getattr(args, "disturbance_channel", "fixed")
    if disturbance_channel_name == "ultrazohm":
        disturbance_channel = UltraZohmDisturbanceChannel(
            can_iface=getattr(args, "uzohm_can_iface", "can0"),
            timeout_s=float(getattr(args, "uzohm_timeout_s", 1.0)),
        )
        disturbance_channel.connect()
    else:
        disturbance_channel = FixedDisturbanceChannel(
            delay_steps=int(getattr(args, "delay_steps", 0)),
            noise_std=float(getattr(args, "noise_std", 0.0)),
            seed=int(getattr(args, "seed", 0)),
        )
    disturbance_channel.reset()
    iter_idx = 0
    last_uzohm_timeout_warning_s = 0.0

    try:
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
                controller_command = [
                    float(leader) + float(args.controller_coeff) * TELEOP_RESIDUAL_ACTION_SCALE * float(residual)
                    for leader, residual in zip(live_obs.leader_joint_pos, residual_action)
                ]
            else:
                controller_command = blend_with_leader(
                    live_obs.leader_joint_pos,
                    controller_action,
                    float(args.controller_coeff),
                )
            if disturbance_channel_name == "ultrazohm":
                raw_commanded_joint_pos = clamp_joint_positions(
                    controller_command, lower_limits, upper_limits)
                raw_follower_action = build_follower_action(
                    raw_commanded_joint_pos)
                try:
                    manipulated_action = disturbance_channel.apply(
                        raw_follower_action)
                    commanded_joint_pos = clamp_joint_positions(
                        hardware_obs_to_joint_positions(manipulated_action),
                        lower_limits,
                        upper_limits,
                    )
                except TimeoutError as exc:
                    now = time_fn()
                    if (now - last_uzohm_timeout_warning_s) >= 2.0:
                        print(f"[WARN] UltraZohm timeout; using raw command this step: {exc}")
                        last_uzohm_timeout_warning_s = now
                    commanded_joint_pos = raw_commanded_joint_pos
            else:
                disturbed_action = disturbance_channel.apply(
                    controller_command)
                commanded_joint_pos = clamp_joint_positions(
                    disturbed_action, lower_limits, upper_limits)
            follower_action = build_follower_action(
                commanded_joint_pos,
                active_joint_names=active_follower_joint_names,
            )
            follower.send_action(follower_action)
            observation_builder.set_previous_action(controller_command)

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
    finally:
        close_method = getattr(disturbance_channel, "close", None)
        if callable(close_method):
            close_method()

    return iter_idx
