"""Lightweight runtime helpers adapted from the reference repo."""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np


def cuda_is_healthy() -> tuple[bool, str]:
    """Best-effort CUDA health check."""

    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() returned False"
        _ = torch.cuda.device_count()
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            if smi.returncode != 0 or not smi.stdout.strip():
                reason = smi.stderr.strip() or smi.stdout.strip() or f"exit code {smi.returncode}"
                return False, f"nvidia-smi check failed: {reason}"
        except FileNotFoundError:
            pass
        return True, "ok"
    except Exception as exc:  # pragma: no cover - runtime specific
        return False, str(exc)


def normalize_device_for_runtime(requested_device: str | None, wants_video: bool = False) -> tuple[str, bool]:
    """Pick a safe runtime device and disable video on CPU fallback."""

    device = requested_device or "cuda:0"
    video_enabled = wants_video
    if device.startswith("cuda"):
        healthy, reason = cuda_is_healthy()
        if not healthy:
            print(f"[WARN] CUDA is unavailable/unhealthy ({reason}). Falling back to CPU.")
            device = "cpu"
            if video_enabled:
                print("[WARN] Disabling video because CPU fallback is active.")
                video_enabled = False
    return device, video_enabled


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Convert a dotted version string into an integer tuple."""

    parts = []
    for token in version.strip().split("."):
        try:
            parts.append(int(token))
        except ValueError:
            break
    return tuple(parts)


def _get_nvidia_driver_version() -> str | None:
    """Return the installed NVIDIA driver version when visible via `nvidia-smi`."""

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if smi.returncode != 0:
        return None
    lines = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def apply_video_renderer_fallback(args_cli: Any, min_rtx_driver: str = "535.129") -> None:
    """Switch to PXR renderer on older driver stacks when video is requested."""

    if not getattr(args_cli, "video", False):
        return

    detected = _get_nvidia_driver_version()
    if detected is None:
        return

    detected_t = _parse_version_tuple(detected)
    minimum_t = _parse_version_tuple(min_rtx_driver)
    if not detected_t or not minimum_t or detected_t >= minimum_t:
        return

    fallback_flags = "--/renderer/multiGpu/enabled=false --/renderer/enabled=pxr --/renderer/active=pxr"
    existing = (getattr(args_cli, "kit_args", "") or "").strip()
    if fallback_flags not in existing:
        args_cli.kit_args = f"{existing} {fallback_flags}".strip()
    print(
        "[WARN] Detected NVIDIA driver "
        f"{detected} < {min_rtx_driver}. "
        "Forcing PXR renderer fallback for video capture."
    )


def _extract_rgb_frame(frame: Any) -> np.ndarray | None:
    """Convert render output into an HxWx3 RGB array when possible."""

    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    return arr[..., :3]


def validate_rgb_rendering(env: Any, max_checks: int = 8) -> tuple[bool, str]:
    """Best-effort probe to detect black/invalid render outputs before recording."""

    try:
        env.reset()
    except Exception:
        pass

    for idx in range(max_checks):
        try:
            frame = env.render()
        except Exception as exc:
            return False, f"env.render() failed: {exc}"

        rgb = _extract_rgb_frame(frame)
        if rgb is not None and float(rgb.max()) > 2.0 and float(rgb.std()) > 1e-3:
            return True, f"valid frame found at check {idx + 1}"

        try:
            action = env.action_space.sample()
            env.step(action)
        except Exception:
            break

    return False, "rendered frames remained black/invalid during warm-up"
