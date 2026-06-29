import sys
from pathlib import Path

# Add project root to PYTHONPATH so "src" can be imported
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if SRC not in sys.path:
    sys.path.insert(0, str(SRC))
