# -*- coding: utf-8 -*-
"""太易 · 紫梅调度控制台 —— 一键启动脚本。

    python launch_zimei_gui.py [--port 8771] [--no-open]
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from zimei_app.server import main  # noqa: E402

if __name__ == "__main__":
    main()
