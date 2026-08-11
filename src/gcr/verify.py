"""Checksum verification for the released artifacts.

``SHA256SUMS`` lists the planner sources, the aggregate benchmark CSVs, the
retained MuJoCo error files, and the retained videos. This is the
cross-platform replacement for the PowerShell snippet the release used before.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gcr.paths import repo_root

__all__ = ["read_manifest", "sha256", "main"]

MANIFEST_NAME = "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest: Path) -> list[tuple[str, str]]:
    """Return ``(digest, relative_path)`` pairs from a ``sha256sum`` style file."""
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(f"{manifest}:{line_number}: cannot parse manifest line: {line!r}")
        digest, relative = parts
        entries.append((digest.lower(), relative.lstrip("*").strip()))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gcr verify",
        description="Verify the checksums recorded in SHA256SUMS.",
    )
    parser.add_argument("--manifest", default=None, help=f"Manifest path. Default: <repo>/{MANIFEST_NAME}")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute and rewrite the digests of the files already listed, instead of checking them.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    manifest = Path(args.manifest) if args.manifest else root / MANIFEST_NAME
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    entries = read_manifest(manifest)

    if args.update:
        lines = []
        for _, relative in entries:
            path = root / relative
            if not path.is_file():
                raise SystemExit(f"Cannot update: listed file is missing: {relative}")
            lines.append(f"{sha256(path)}  {relative}")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Rewrote {len(lines)} digests in {manifest}")
        return 0

    missing: list[str] = []
    mismatched: list[str] = []
    for expected, relative in entries:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256(path) != expected:
            mismatched.append(relative)

    for relative in missing:
        print(f"MISSING   {relative}")
    for relative in mismatched:
        print(f"MISMATCH  {relative}")

    checked = len(entries) - len(missing)
    if missing or mismatched:
        print(f"{len(entries) - len(missing) - len(mismatched)}/{len(entries)} files verified.")
        return 1
    print(f"All {checked} files match {manifest.name}.")
    return 0
