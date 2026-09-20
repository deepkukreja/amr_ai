from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "datasets/torwic_segmentation/split"
)

MODEL_NAME = "nvidia/mit-b0"

OUTPUT_DIR = Path(
    "runs/segformer_torwic_targeted"
)

NUM_CLASSES = 16

IMAGE_SIZE = 640
BATCH_SIZE = 1
EPOCHS = 10

LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# CLASS NAMES
# ============================================================

ID2LABEL = {
    0: "background",
    1: "driveable_ground",
    2: "ceiling",
    3: "ego_vehicle",
    4: "wall_fence_pillar",
    5: "misc_static_feature",
    6: "rack_shelf",
    7: "goods_material",
    8: "fixed_machinery",
    9: "cart_pallet_jack",
    10: "pylon_cone",
    11: "text_region",
    12: "misc_non_static_feature",
    13: "person",
    14: "fork_truck",
    15: "misc_dynamic_feature",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

class TorWICDataset(Dataset):

    def __init__(self, root, split, processor):

        self.root = Path(root)
        self.split = split
        self.processor = processor

        self.image_dir = (
            self.root / split / "images"
        )

        self.mask_dir = (
            self.root / split / "masks"
        )

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

        self.masks = [
            self.mask_dir / image_path.name
            for image_path in self.images
        ]

        print(
            f"{split.capitalize()} images: "
            f"{len(self.images)}"
        )


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        # ----------------------------------------------------
        # Resize image to 640x640
        # ----------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Resize mask to 640x640
        #
        # NEAREST is required because mask values are class IDs.
        # ----------------------------------------------------

        mask = mask.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.NEAREST
        )

        encoded = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = (
            encoded["pixel_values"]
            .squeeze(0)
        )

        labels = torch.tensor(
            np.array(
                mask,
                dtype=np.int64
            ),
            dtype=torch.long
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# ============================================================
# PROCESSOR
# ============================================================

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME
)

processor.size = {
    "height": IMAGE_SIZE,
    "width": IMAGE_SIZE,
}


# ============================================================
# DATASETS
# ============================================================

train_dataset = TorWICDataset(
    DATASET_ROOT,
    "train",
    processor
)

val_dataset = TorWICDataset(
    DATASET_ROOT,
    "val",
    processor
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# TARGETED OPERATIONAL CLASS WEIGHTS
# ============================================================
#
# These are NOT based on pixel frequency.
#
# Higher values tell the loss:
# "Mistakes on this class matter more."
#
# We deliberately give greater importance to classes that are
# important for an autonomous warehouse robot.
# ============================================================

targeted_weights = np.array(
    [
        0.50,  # 0  background
        1.00,  # 1  driveable_ground
        0.70,  # 2  ceiling
        1.00,  # 3  ego_vehicle
        0.70,  # 4  wall_fence_pillar
        0.70,  # 5  misc_static_feature

        2.50,  # 6  rack_shelf

        0.90,  # 7  goods_material
        1.50,  # 8  fixed_machinery

        3.00,  # 9  cart_pallet_jack
        1.50,  # 10 pylon_cone
        0.50,  # 11 text_region

        1.50,  # 12 misc_non_static_feature
        2.00,  # 13 person
        2.50,  # 14 fork_truck
        2.00,  # 15 misc_dynamic_feature
    ],
    dtype=np.float32
)


# Normalize so the average weight is approximately 1.
targeted_weights = (
    targeted_weights
    / targeted_weights.mean()
)


print()
print("=" * 72)
print("TARGETED OPERATIONAL CLASS WEIGHTS")
print("=" * 72)

for class_id in range(NUM_CLASSES):

    print(
        f"{class_id:>2}  "
        f"{ID2LABEL[class_id]:<28} "
        f"weight: "
        f"{targeted_weights[class_id]:.4f}"
    )

print("=" * 72)


class_weights = torch.tensor(
    targeted_weights,
    dtype=torch.float32,
    device=device
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading pretrained SegFormer...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id={
        name: idx
        for idx, name in ID2LABEL.items()
    },
    ignore_mismatched_sizes=True
)

model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

best_val_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0.0

    for batch in train_loader:

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            pixel_values=pixel_values
        )

        # SegFormer produces lower-resolution logits.
        target_labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=outputs.logits.shape[-2:],
            mode="nearest"
        ).squeeze(1).long()

        loss = F.cross_entropy(
            outputs.logits,
            target_labels,
            weight=class_weights
        )

        loss.backward()

        optimizer.step()

        train_loss_total += loss.item()


    train_loss = (
        train_loss_total /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    with torch.no_grad():

        for batch in val_loader:

            pixel_values = batch[
                "pixel_values"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                pixel_values=pixel_values
            )

            target_labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=outputs.logits.shape[-2:],
                mode="nearest"
            ).squeeze(1).long()

            loss = F.cross_entropy(
                outputs.logits,
                target_labels,
                weight=class_weights
            )

            val_loss_total += loss.item()


    val_loss = (
        val_loss_total /
        len(val_loader)
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_dir = (
            OUTPUT_DIR / "best"
        )

        best_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(
            best_dir
        )

        processor.save_pretrained(
            best_dir
        )

        print(
            f"  ✓ New best model saved "
            f"(val loss = {val_loss:.4f})"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 72)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best model saved to: "
    f"{OUTPUT_DIR / 'best'}"
)

print("=" * 72)
