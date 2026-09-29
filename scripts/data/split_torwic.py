from pathlib import Path
import random
import shutil

ROOT = Path("torwic_segmentation")
IMAGES = ROOT / "images"
MASKS = ROOT / "masks"

OUTPUT = ROOT / "split"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

# Each output filename looks like:
# 000001751_image_0.png
# The first part is the original sequence ID.
images = sorted(IMAGES.glob("*.png"))

sequences = sorted({
    image.stem.rsplit("_image_", 1)[0]
    for image in images
})

print(f"Total images: {len(images)}")
print(f"Total sequences: {len(sequences)}")

random.shuffle(sequences)

n = len(sequences)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_sequences = sequences[:n_train]
val_sequences = sequences[n_train:n_train + n_val]
test_sequences = sequences[n_train + n_val:]

splits = {
    "train": train_sequences,
    "val": val_sequences,
    "test": test_sequences,
}

for split_name, split_sequences in splits.items():
    (OUTPUT / split_name / "images").mkdir(
        parents=True,
        exist_ok=True,
    )
    (OUTPUT / split_name / "masks").mkdir(
        parents=True,
        exist_ok=True,
    )

sequence_sets = {
    name: set(values)
    for name, values in splits.items()
}

counts = {
    "train": 0,
    "val": 0,
    "test": 0,
}

for image_path in images:

    sequence = image_path.stem.rsplit("_image_", 1)[0]

    if sequence in sequence_sets["train"]:
        split = "train"
    elif sequence in sequence_sets["val"]:
        split = "val"
    elif sequence in sequence_sets["test"]:
        split = "test"
    else:
        raise RuntimeError(
            f"Sequence {sequence} not assigned to a split"
        )

    mask_path = MASKS / image_path.name

    if not mask_path.exists():
        raise RuntimeError(
            f"Missing mask for {image_path.name}"
        )

    shutil.copy2(
        image_path,
        OUTPUT / split / "images" / image_path.name,
    )

    shutil.copy2(
        mask_path,
        OUTPUT / split / "masks" / mask_path.name,
    )

    counts[split] += 1

print()
print("==============================")
print("TorWIC split complete")
print("==============================")
print(f"Train sequences: {len(train_sequences)}")
print(f"Val sequences:   {len(val_sequences)}")
print(f"Test sequences:  {len(test_sequences)}")
print()
print(f"Train images: {counts['train']}")
print(f"Val images:   {counts['val']}")
print(f"Test images:  {counts['test']}")
