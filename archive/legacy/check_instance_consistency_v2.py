import json
from pathlib import Path
from collections import Counter

root = Path("datasets/torwic_semantic/Labelled_Test_Data")

classes = [
    "person",
    "fork_truck",
    "cart_pallet_jack",
    "rack_shelf",
    "misc_dynamic_feature",
]

json_files = sorted(root.rglob("raw_labels_2d.json"))

print("=" * 75)
print("TORWIC INSTANCE-FILE CONSISTENCY CHECK v2")
print("=" * 75)
print(f"Annotation JSON files: {len(json_files)}")

expected_totals = Counter()
actual_totals = Counter()
mismatches = []


def expected_instance_count(value):
    """
    Count actual annotated instances.

    Lists:
        each element represents an instance.

    Dictionaries:
        only count if numPixels > 0.
        A dict with numPixels == 0 means the class is absent
        from that image.
    """
    if value is None:
        return 0

    if isinstance(value, list):
        return len(value)

    if isinstance(value, dict):
        return 1 if value.get("numPixels", 0) > 0 else 0

    return 0


for json_file in json_files:

    with open(json_file, "r") as f:
        data = json.load(f)

    mapping = data.get("labelMapping", {})
    folder = json_file.parent

    for cls in classes:

        entry = mapping.get(cls)

        expected = expected_instance_count(entry)

        # Search for the corresponding PNG files.
        png_files = []

        plain = folder / f"{cls}.png"
        if plain.exists():
            png_files.append(plain)

        png_files.extend(sorted(folder.glob(f"{cls}_*.png")))

        actual = len(png_files)

        expected_totals[cls] += expected
        actual_totals[cls] += actual

        if expected != actual:
            mismatches.append(
                {
                    "folder": str(folder),
                    "class": cls,
                    "expected": expected,
                    "actual": actual,
                    "files": [p.name for p in png_files],
                    "mapping": entry,
                }
            )


print("\nAGGREGATE COUNTS")
print("-" * 75)

for cls in classes:
    print(
        f"{cls:30s}"
        f" expected={expected_totals[cls]:4d}"
        f"  PNG files={actual_totals[cls]:4d}"
    )

print("\nMISMATCHES:", len(mismatches))

if mismatches:

    print("\nFIRST 30 MISMATCHES")
    print("-" * 75)

    for m in mismatches[:30]:
        print(f"\nClass:     {m['class']}")
        print(f"Folder:    {m['folder']}")
        print(f"Expected:  {m['expected']}")
        print(f"Actual:    {m['actual']}")
        print(f"Files:     {m['files']}")
        print(f"Mapping:   {m['mapping']}")

else:

    print("\nSUCCESS:")
    print("Every checked object class has the expected PNG representation.")
