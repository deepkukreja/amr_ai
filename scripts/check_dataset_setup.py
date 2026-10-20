#!/usr/bin/env python3
"""Read-only audit of dataset YAML files, manifests, and image availability."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def split_values(value):
    """Normalize a dataset split into a list of path strings."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []

    return [
        item.strip()
        for item in values
        if isinstance(item, str) and item.strip()
    ]


def resolve_config_base(data, yaml_file, repo_root):
    """Determine the base directory used by a dataset configuration."""
    declared_path = data.get("path")

    if not isinstance(declared_path, str) or not declared_path.strip():
        return yaml_file.parent.resolve()

    path = Path(declared_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    # Relative 'path:' values are interpreted from the repository root.
    return (repo_root / path).resolve()


def resolve_target(value, base):
    """Resolve a dataset path without requiring it to exist."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (base / path).resolve()


def read_manifest(manifest):
    """Return image paths listed in a text manifest and any read error."""
    try:
        lines = manifest.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        return [], str(exc)

    references = []

    for line in lines:
        entry = line.strip()

        if not entry or entry.startswith("#"):
            continue

        path = Path(entry).expanduser()

        if not path.is_absolute():
            path = manifest.parent / path

        references.append(path.resolve())

    return references, None


def count_direct_images(directory):
    """Count image files directly inside a directory, without recursion."""
    try:
        return sum(
            1
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    except OSError:
        return 0


def display_path(path, root):
    """Prefer repository-relative paths in the report."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def discover_dataset_configs(repo_root):
    """Find parseable dataset YAML files and collect configuration errors."""
    configs = []
    errors = []
    calibration_files_skipped = 0

    search_dirs = [
        repo_root / "configs",
        repo_root / "datasets",
    ]

    yaml_files = []

    for directory in search_dirs:
        if directory.is_dir():
            yaml_files.extend(directory.rglob("*.yaml"))
            yaml_files.extend(directory.rglob("*.yml"))

    for yaml_file in sorted(set(yaml_files)):
        if "calibration_files" in yaml_file.parts:
            calibration_files_skipped += 1
            continue

        try:
            data = yaml.safe_load(
                yaml_file.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            errors.append(
                f"{display_path(yaml_file, repo_root)}: {exc}"
            )
            continue

        if not isinstance(data, dict):
            continue

        # A dataset YAML must define at least one recognized split.
        if not any(split in data for split in SPLITS):
            continue

        configs.append({
            "file": yaml_file,
            "data": data,
            "base": resolve_config_base(
                data, yaml_file, repo_root
            ),
        })

    return configs, errors, calibration_files_skipped


def inspect_split(value, base, repo_root):
    """Inspect a split path and report missing data without changing files."""
    findings = []

    for raw_path in split_values(value):
        target = resolve_target(raw_path, base)

        finding = {
            "target": display_path(target, repo_root),
            "state": "",
        }

        if not target.exists():
            is_repo_target = target == repo_root or repo_root in target.parents
            finding["state"] = (
                "MISSING TEMP PATH"
                if str(target).startswith("/tmp/") and not is_repo_target
                else "MISSING TARGET"
            )

        elif target.is_file() and target.suffix.lower() == ".txt":
            references, error = read_manifest(target)

            if error:
                finding["state"] = "UNREADABLE MANIFEST"
                finding["detail"] = error
            else:
                missing = [
                    image_path
                    for image_path in references
                    if not image_path.is_file()
                ]

                finding["state"] = (
                    "OK" if not missing else "MISSING IMAGES"
                )
                finding["references"] = len(references)
                finding["missing_images"] = len(missing)
                finding["missing_examples"] = [
                    display_path(path, repo_root)
                    for path in missing[:3]
                ]

        elif target.is_dir():
            image_count = count_direct_images(target)
            finding["state"] = (
                "OK" if image_count else "EMPTY DIRECTORY"
            )
            finding["images"] = image_count

        elif target.is_file():
            finding["state"] = "OK"

        else:
            finding["state"] = "UNUSABLE TARGET"

        findings.append(finding)

    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Audit dataset configuration and data availability."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (defaults to this script's repository).",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Also display paths that passed their checks.",
    )
    args = parser.parse_args()

    repo_root = (
        args.root.expanduser().resolve()
        if args.root
        else Path(__file__).resolve().parents[1]
    )

    if not repo_root.is_dir():
        parser.error(f"Repository directory does not exist: {repo_root}")

    configs, errors, calibration_skipped = (
        discover_dataset_configs(repo_root)
    )

    totals = {
        "targets": 0,
        "ok": 0,
        "missing_targets": 0,
        "missing_temp_paths": 0,
        "missing_images": 0,
        "image_references": 0,
        "empty_directories": 0,
        "unreadable_manifests": 0,
        "unusable_targets": 0,
    }

    print(f"Repository: {repo_root}")
    print(f"Dataset YAMLs discovered: {len(configs)}")
    print(f"Calibration YAMLs skipped: {calibration_skipped}")

    for config in configs:
        yaml_file = config["file"]
        data = config["data"]
        base = config["base"]
        resolved_splits = {}

        print(f"\n{display_path(yaml_file, repo_root)}")

        for split in SPLITS:
            if split not in data:
                continue

            values = split_values(data[split])
            resolved_splits[split] = tuple(
                resolve_target(value, base) for value in values
            )

            if not values:
                print(f"  [EMPTY SPLIT] {split}")
                continue

            findings = inspect_split(data[split], base, repo_root)

            for finding in findings:
                state = finding["state"]
                totals["targets"] += 1

                if state == "OK":
                    totals["ok"] += 1
                elif state == "MISSING TARGET":
                    totals["missing_targets"] += 1
                elif state == "MISSING TEMP PATH":
                    totals["missing_temp_paths"] += 1
                elif state == "MISSING IMAGES":
                    totals["missing_images"] += finding.get(
                        "missing_images", 0
                    )
                    totals["image_references"] += finding.get(
                        "references", 0
                    )
                elif state == "EMPTY DIRECTORY":
                    totals["empty_directories"] += 1
                elif state == "UNREADABLE MANIFEST":
                    totals["unreadable_manifests"] += 1
                elif state == "UNUSABLE TARGET":
                    totals["unusable_targets"] += 1

                if state != "OK" or args.show_ok:
                    print(
                        f"  [{state}] {split}: {finding['target']}"
                    )

                for example in finding.get("missing_examples", []):
                    print(f"      Missing example: {example}")

        if (
            resolved_splits.get("val")
            and resolved_splits.get("val") == resolved_splits.get("test")
        ):
            print(
                "  [NOTE] val and test resolve to the same path(s); "
                "they are not independent evaluation sets."
            )

    print("\n========== SUMMARY ==========")
    print(f"Split targets inspected: {totals['targets']}")
    print(f"Targets passing checks: {totals['ok']}")
    print(f"Missing non-temporary targets: {totals['missing_targets']}")
    print(f"Missing temporary paths under /tmp: {totals['missing_temp_paths']}")
    print(f"Empty image directories: {totals['empty_directories']}")
    print(f"Unreadable manifests: {totals['unreadable_manifests']}")
    print(f"Unusable targets: {totals['unusable_targets']}")
    print(f"Image references in affected manifests: {totals['image_references']}")
    print(f"Missing referenced images: {totals['missing_images']}")
    print(f"Configuration read/parse errors: {len(errors)}")

    for error in errors:
        print(f"  [CONFIG ERROR] {error}")

    print("\nAudit complete. No dataset files were modified.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
