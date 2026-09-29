from pathlib import Path

import torch
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

NUM_CLASSES = 16
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 6e-5

OUTPUT_DIR = Path("runs/segformer_torwic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

LABEL2ID = {
    name: idx
    for idx, name in ID2LABEL.items()
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATASET
# ============================================================

class TorWICDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        processor,
    ):
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

        if not self.images:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        for image_path in self.images:

            mask_path = (
                self.mask_dir /
                image_path.name
            )

            if not mask_path.exists():
                raise RuntimeError(
                    f"Missing mask: {mask_path}"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = (
            self.mask_dir /
            image_path.name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            do_reduce_labels=False,
            return_tensors="pt",
        )

        encoded["pixel_values"] = (
            encoded["pixel_values"].squeeze(0)
        )

        encoded["labels"] = (
            encoded["labels"]
            .squeeze(0)
            .long()
        )

        return encoded


# ============================================================
# PROCESSOR
# ============================================================

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME,
    do_reduce_labels=False,
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = TorWICDataset(
    DATASET_ROOT,
    "train",
    processor,
)

val_dataset = TorWICDataset(
    DATASET_ROOT,
    "val",
    processor,
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# MODEL
# ============================================================

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    print()
    print(
        "=" * 60
    )
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0.0

    for step, batch in enumerate(
        train_loader
    ):

        pixel_values = (
            batch["pixel_values"]
            .to(device)
        )

        labels = (
            batch["labels"]
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item()
        )

        if (
            step + 1
        ) % 20 == 0:

            print(
                f"Step {step + 1}/"
                f"{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

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

            pixel_values = (
                batch["pixel_values"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            val_loss_total += (
                outputs.loss.item()
            )

    val_loss = (
        val_loss_total /
        len(val_loader)
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_dir = (
            OUTPUT_DIR /
            "best"
        )

        best_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            best_dir
        )

        processor.save_pretrained(
            best_dir
        )

        print(
            "Saved new best model."
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Best model:",
    OUTPUT_DIR / "best"
)
