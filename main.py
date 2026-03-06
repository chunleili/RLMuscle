import importlib
import os
from pathlib import Path

from dotenv import load_dotenv


def _print_available_runs(example_stems: list[str]) -> None:
    print("Available RUN values:")
    for stem in example_stems:
        print(f"  - {stem}")


def main() -> int:
    load_dotenv()

    examples_dir = Path(__file__).resolve().parent / "examples"
    example_stems = sorted(path.stem for path in examples_dir.glob("example_*.py") if path.is_file())
    run_key = os.environ.get("RUN", "").strip()

    if not run_key:
        print("RUN is not set.")
        _print_available_runs(example_stems)
        return 1

    if run_key not in example_stems:
        print(f"Unknown RUN value: {run_key}")
        _print_available_runs(example_stems)
        return 1

    print(f"Running example: {run_key}")
    module = importlib.import_module(f"examples.{run_key}")
    if hasattr(module, "main"):
        module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
