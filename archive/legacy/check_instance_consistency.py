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
print("TORWIC INSTANCE-FILE CONSISTENCY CHECK")
print("=" * 75)
print(f"Annotation JSON files: {len(json_files)}")

expected_totals = Counter()
actual_totals = Counter()
mismatches = []

for json_file in json_files:

    with open(json_file, "r") as f:
        data = json.load(f)

    mapping = data.get("labelMapping", {})
    folder = json_file.parent

    for cls in classes:

        entry = mapping.get(cls)

        if entry is None:
            expected = 0
        elif isinstance(entry, list):
            expected = len(entry)
        elif isinstance(entry, dict):
            expected = 1
        else:
            expected = 0

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
                (
                    folder,
                    cls,
                    expected,
                    actual,
                    [p.name for p in png_files],
                )
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

    print("\nFIRST 20 MISMATCHES")
    print("-" * 75)

    for folder, cls, expected, actual, files in mismatches[:20]:

        print(f"\nClass:     {cls}")
        print(f"Folder:    {folder}")
        print(f"Expected:  {expected}")
        print(f"Actual:    {actual}")
        print(f"Files:     {files}")

else:

    print("\nNo mismatches found.")
    print("The instance PNGs consistently represent the labelMapping entries.")
