from __future__ import annotations

import os
import sys


argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
if "pytest" in argv0:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
