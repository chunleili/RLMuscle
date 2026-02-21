"""Shared pytest configuration for the test suite."""
import sys
from pathlib import Path

# Allow imports like `from VMuscle.muscle_warp import ...` from test files.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
