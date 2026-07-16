"""Pytest configuration: ensure `src/` is importable.

This file is loaded by pytest before any tests run. It adds the `src/`
directory to `sys.path` so that `import src.data_loader` works without
needing to install the package as editable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
