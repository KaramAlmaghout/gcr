"""Repository path resolution.

Every module that needs a checked-in asset (the vendored MuJoCo Menagerie
UR10e, the paper benchmark snapshot, the C++ benchmark binary) resolves it
through this module instead of counting ``Path(__file__).parents[...]`` levels,
so moving a module inside the package cannot silently break asset lookup.

Resolution order for the repository root:

1. the ``GCR_REPO_ROOT`` environment variable;
2. the first directory at or above this file that looks like the repository;
3. the first directory at or above the current working directory that does.

Step 3 covers a non-editable install, where the package no longer lives inside
the checkout; in that case run the commands from the repository, or set
``GCR_REPO_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "results_dir",
    "default_benchmark_dir",
    "outlier_analysis_dir",
    "third_party_dir",
    "cpp_dir",
    "cpp_binary",
]

_ENV_VAR = "GCR_REPO_ROOT"


def _looks_like_repo(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "gcr").is_dir()


def _search_upwards(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if _looks_like_repo(candidate):
            return candidate
    return None


def repo_root() -> Path:
    """Return the repository root directory."""
    env = os.environ.get(_ENV_VAR)
    if env:
        candidate = Path(env).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"{_ENV_VAR}='{env}' but that path does not exist")
        return candidate

    found = _search_upwards(Path(__file__).resolve().parent) or _search_upwards(Path.cwd().resolve())
    if found is not None:
        return found

    raise FileNotFoundError(
        "Could not locate the GCR repository root. Run the command from inside "
        f"the repository, or set {_ENV_VAR} to its path."
    )


def results_dir() -> Path:
    """Directory holding the checked-in result artifacts."""
    return repo_root() / "results"


def default_benchmark_dir() -> Path:
    """The 1000-run paper benchmark snapshot."""
    return results_dir() / "paper_benchmark_1000"


def outlier_analysis_dir() -> Path:
    """Directory holding the runtime-outlier sensitivity analysis."""
    return results_dir() / "paper_report" / "outlier_analysis"


def third_party_dir() -> Path:
    """Vendored third-party assets."""
    return repo_root() / "third_party"


def cpp_dir() -> Path:
    """C++ benchmark sources and build files."""
    return repo_root() / "cpp"


def cpp_binary() -> Path | None:
    """Return the built C++ benchmark binary, or ``None`` when it is missing."""
    build_dir = cpp_dir() / "build"
    candidates = (
        build_dir / "gcr_benchmark.exe",
        build_dir / "gcr_benchmark",
        build_dir / "Release" / "gcr_benchmark.exe",
        build_dir / "Debug" / "gcr_benchmark.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
