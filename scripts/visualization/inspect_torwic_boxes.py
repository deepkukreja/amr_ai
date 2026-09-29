from pathlib import Path
import json


# ============================================================
# FIND A TORWIC 2D LABEL FILE
# ============================================================

SEARCH_ROOTS = [
    Path("datasets/torwic_semantic"),
    Path("datasets"),
]

json_files = []

for root in SEARCH_ROOTS:

    if root.exists():

        json_files.extend(
            root.rglob("raw_labels_2d.json")
        )


# Remove duplicates
json_files = sorted(
    set(json_files)
)


print()
print("=" * 72)
print("TORWIC 2D ANNOTATION INSPECTION")
print("=" * 72)

print(
    f"Found annotation files: {len(json_files)}"
)


if not json_files:

    print()
    print("No raw_labels_2d.json files were found.")
    print()
    print("Search the dataset location manually with:")
    print(
        "find datasets -name 'raw_labels_2d.json' | head"
    )
    raise SystemExit


# ============================================================
# SELECT FIRST FILE
# ============================================================

annotation_path = json_files[0]

print()
print("Using:")
print(annotation_path)


# ============================================================
# LOAD JSON
# ============================================================

with open(
    annotation_path,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ============================================================
# TOP-LEVEL STRUCTURE
# ============================================================

print()
print("-" * 72)
print("TOP-LEVEL TYPE")
print("-" * 72)

print(
    type(data).__name__
)


if isinstance(data, dict):

    print()
    print("Top-level keys:")

    for key in data.keys():

        print(
            f"  - {key}"
        )


elif isinstance(data, list):

    print(
        f"Number of top-level items: {len(data)}"
    )


# ============================================================
# PRINT A SAFE STRUCTURAL SUMMARY
# ============================================================

def summarize(value, name="root", depth=0):

    indent = "  " * depth

    if depth > 3:
        return

    if isinstance(value, dict):

        print(
            f"{indent}{name}: dict "
            f"({len(value)} keys)"
        )

        for i, (key, child) in enumerate(
            value.items()
        ):

            if i >= 15:
                print(
                    f"{indent}  ..."
                )
                break

            summarize(
                child,
                str(key),
                depth + 1
            )

    elif isinstance(value, list):

        print(
            f"{indent}{name}: list "
            f"({len(value)} items)"
        )

        if len(value) > 0:

            summarize(
                value[0],
                "[0]",
                depth + 1
            )

    else:

        print(
            f"{indent}{name}: "
            f"{type(value).__name__} = "
            f"{repr(value)[:150]}"
        )


print()
print("-" * 72)
print("STRUCTURE")
print("-" * 72)

summarize(data)


# ============================================================
# PRINT SMALL SAMPLE
# ============================================================

print()
print("=" * 72)
print("SMALL SAMPLE")
print("=" * 72)

text = json.dumps(
    data,
    indent=2,
    ensure_ascii=False
)

# Only show first 5000 characters
print(
    text[:5000]
)

if len(text) > 5000:

    print()
    print("... sample truncated ...")

print("=" * 72)
