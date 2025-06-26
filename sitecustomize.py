"""Site customizations for OpenCam development.
Automatically adds the `python/` subdirectory to `sys.path` so that the
`import opencam` works from a fresh source checkout without needing an
editable install.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    # Prepend to ensure it has priority over any globally installed version.
    sys.path.insert(0, str(PYTHON_DIR)) 