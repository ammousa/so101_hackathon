"""UltraZohm deploy-time disturbance channel."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any


DEFAULT_UZOHM_CAN_IFACE = "can0"
DEFAULT_UZOHM_TIMEOUT_S = 1.0


def _ultrazohm_scripts_dir() -> Path:
    """Return the bundled UltraZohm scripts directory."""
    return Path(__file__).resolve().parents[2] / "external" / "ultrazohm" / "sebi-scripts"


@dataclass
class UltraZohmDisturbanceChannel:
    """Route LeRobot action dictionaries through the UltraZohm CAN path."""

    can_iface: str = DEFAULT_UZOHM_CAN_IFACE
    timeout_s: float = DEFAULT_UZOHM_TIMEOUT_S

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        self._uzohm_port: Any | None = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the bundled UltraZohm CAN client."""
        if self._connected:
            return

        scripts_dir = _ultrazohm_scripts_dir()
        if not scripts_dir.exists():
            raise ModuleNotFoundError(f"Could not find UltraZohm scripts at `{scripts_dir}`")
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        self._uzohm_port = importlib.import_module("uzohmPort")
        self._uzohm_port.connect(self.can_iface, timeout_s=float(self.timeout_s))
        self._connected = True

    def reset(self) -> None:
        """Reset channel state."""

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        """Return UltraZohm-manipulated LeRobot action values."""
        if not self._connected:
            self.connect()
        assert self._uzohm_port is not None
        return dict(self._uzohm_port.manipulate(action))

    def close(self) -> None:
        """Close the UltraZohm client."""
        if self._uzohm_port is None:
            return
        self._uzohm_port.close()
        self._connected = False
