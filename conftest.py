from __future__ import annotations

import pytest


class _LaunchTestingCompatHooks:
    @pytest.hookspec
    def pytest_launch_collect_makemodule(self, path, parent, entrypoint):
        """Compatibility hookspec for externally installed ROS launch-testing plugins."""


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(_LaunchTestingCompatHooks)
