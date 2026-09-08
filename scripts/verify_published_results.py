"""Verify the immutable result snapshot without model or network access."""

import gzip
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    seen = set()
    for record in manifest["files"]:
        relative = record["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "results") or relative in seen:
            failures.append(f"Invalid or duplicate path: {relative}")
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append(f"Missing: {relative}")
            continue
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            failures.append(f"Hash or size mismatch: {relative}")
        if record.get("encoding") == "gzip":
            try:
                original = gzip.decompress(data)
                if len(original) != record["source_bytes"] or hashlib.sha256(original).hexdigest() != record["source_sha256"]:
                    failures.append(f"Decompressed content mismatch: {relative}")
            except (OSError, EOFError):
                failures.append(f"Invalid gzip: {relative}")
    if len(seen) != manifest["result_files"]:
        failures.append("Manifest file count mismatch")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(seen):,} published result files and all compressed-source hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
