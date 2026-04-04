"""Shared pytest configuration for the test suite."""
import sys
from pathlib import Path

# Add project root so `from examples.xxx import ...` works in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
