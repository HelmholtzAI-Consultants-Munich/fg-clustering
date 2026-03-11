#!/usr/bin/env python3
"""
Extract supported Python versions from pyproject.toml's
[project].requires-python field and emit a JSON list.

Expected format:
    ">=3.10,<3.14"

Example output:
    ["3.10", "3.11", "3.12", "3.13"]
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> None:
    try:
        import tomllib  # Python >=3.11
    except ModuleNotFoundError:
        raise SystemExit("tomllib not available. This script must run with Python >= 3.11.")

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise SystemExit("pyproject.toml not found in repository root.")

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    try:
        requires_python = data["project"]["requires-python"]
    except KeyError:
        raise SystemExit("Missing 'project.requires-python' in pyproject.toml.")

    # Expect format like: ">=3.10,<3.14"
    min_match = re.search(r">=\s*3\.(\d+)", requires_python)
    max_match = re.search(r"<\s*3\.(\d+)", requires_python)

    if not (min_match and max_match):
        raise SystemExit(
            f"Unsupported requires-python format: {requires_python!r}. "
            "Expected format like '>=3.10,<3.14'."
        )

    min_minor = int(min_match.group(1))
    max_minor = int(max_match.group(1))  # exclusive

    if min_minor >= max_minor:
        raise SystemExit(f"Invalid Python version range: {requires_python!r}")

    versions = [f"3.{minor}" for minor in range(min_minor, max_minor)]

    # Print pure JSON for GitHub Actions consumption
    print(json.dumps(versions))


if __name__ == "__main__":
    main()
