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
    "runs/segformer_torwic_augmented"
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
# TRAINING AUGMENTATION
# ============================================================

def augment_image_and_mask(image, mask):

    # --------------------------------------------------------
    # 1. Random horizontal flip
    # --------------------------------------------------------

    if random.random() < 0.5:

        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        mask = mask.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )


    # --------------------------------------------------------
    # 2. Random resized crop
    #
    # Crop between 60% and 100% of the original area.
    # Then resize the crop to 640x640.
    #
    # This creates zoomed-in training examples and teaches
    # the model to recognize objects at different scales.
    # --------------------------------------------------------

    original_width, original_height = image.size

    original_area = (
        original_width * original_height
    )

    for _ in range(10):

        scale = random.uniform(
            0.60,
            1.00
        )

        target_area = (
            original_area * scale
        )

        crop_ratio = random.uniform(
            1.5,
            2.0
        )

        crop_width = int(
            np.sqrt(
                target_area * crop_ratio
            )
        )

        crop_height = int(
            np.sqrt(
                target_area / crop_ratio
            )
        )

        if (
            crop_width <= original_width
            and
            crop_height <= original_height
        ):

            left = random.randint(
                0,
                original_width - crop_width
            )

            top = random.randint(
                0,
                original_height - crop_height
            )

            right = left + crop_width
            bottom = top + crop_height

            image = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )

            mask = mask.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )

            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.BILINEAR
            )

            mask = mask.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.NEAREST
            )

            return image, mask


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    mask = mask.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.NEAREST
    )

    return image, mask


# ============================================================
# DATASET
# ============================================================

class TorWICDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        processor,
        training=False
    ):

        self.root = Path(root)
        self.split = split
        self.processor = processor
        self.training = training

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
        # Training augmentation
        # ----------------------------------------------------

        if self.training:

            image, mask = (
                augment_image_and_mask(
                    image,
                    mask
                )
            )

        else:

            # Validation: NO RANDOM AUGMENTATION
            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.BILINEAR
            )

            mask = mask.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.NEAREST
            )


        # ----------------------------------------------------
        # Convert image to model tensor
        # ----------------------------------------------------

        encoded = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = (
            encoded["pixel_values"]
            .squeeze(0)
        )


        # ----------------------------------------------------
        # Convert mask to tensor
        # ----------------------------------------------------

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
    processor,
    training=True
)

val_dataset = TorWICDataset(
    DATASET_ROOT,
    "val",
    processor,
    training=False
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
# CLASS PIXEL FREQUENCIES
# ============================================================

print()
print("Calculating class frequencies...")

class_pixel_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

for mask_path in train_dataset.masks:

    mask = np.array(
        Image.open(
            mask_path
        ).convert("L"),
        dtype=np.int64
    )

    counts = np.bincount(
        mask.reshape(-1),
        minlength=NUM_CLASSES
    )

    class_pixel_counts += counts


total_pixels = (
    class_pixel_counts.sum()
)

class_frequencies = (
    class_pixel_counts /
    total_pixels
)


# ============================================================
# SAME CLASS WEIGHT METHOD AS PREVIOUS EXPERIMENT
# ============================================================

weights = np.zeros(
    NUM_CLASSES,
    dtype=np.float32
)

for class_id in range(NUM_CLASSES):

    freq = class_frequencies[class_id]

    if freq > 0:

        weights[class_id] = (
            1.0 / np.sqrt(freq)
        )

    else:

        weights[class_id] = 0.0


nonzero = weights > 0

weights[nonzero] = (
    weights[nonzero]
    / weights[nonzero].mean()
)

weights = np.clip(
    weights,
    0.5,
    5.0
)


print()
print("=" * 72)
print("CLASS WEIGHTS")
print("=" * 72)

for class_id in range(NUM_CLASSES):

    print(
        f"{class_id:>2}  "
        f"{ID2LABEL[class_id]:<28}  "
        f"pixels: "
        f"{int(class_pixel_counts[class_id]):>12}  "
        f"weight: "
        f"{weights[class_id]:.4f}"
    )

print("=" * 72)


class_weights = torch.tensor(
    weights,
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
# OUTPUT
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


    # ========================================================
    # VALIDATION
    # ========================================================

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


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

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
