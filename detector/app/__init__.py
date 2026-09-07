"""Makes the sibling top-level `fusion/` package (repo_root/fusion) importable
from inside `app`, without changing how this service is run or how any other
module here imports `app` itself (still `from app import ...`, cwd still
`detector/`, per detector/README.md). Runs once, on first `import app`."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
