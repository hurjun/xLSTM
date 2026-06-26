"""Make the repository root importable so ``import xlstm`` works under pytest.

A root-level ``conftest.py`` causes pytest to prepend the repo root to
``sys.path`` during collection, so the tests can import the in-repo ``xlstm``
package without an editable install.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
