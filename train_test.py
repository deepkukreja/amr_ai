from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = Path(
    "datasets/torwic_segmentation/split"
)

MODEL_NAME = "nvidia/mit-b0"
NUM_CLASSES = 16

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


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class TorWICDataset(Dataset):

    def __init__(self, root, split, processor):
        self.root = Path(root)
        self.split = split
        self.processor = processor

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        for image_path in self.images:
            mask_path = self.mask_dir / image_path.name

            if not mask_path.exists():
                raise RuntimeError(
                    f"Missing mask for {image_path.name}"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name

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

        # Remove batch dimension added by processor
        encoded["pixel_values"] = (
            encoded["pixel_values"].squeeze(0)
        )

        encoded["labels"] = (
            encoded["labels"].squeeze(0).long()
        )

        return encoded


# --------------------------------------------------
# Processor
# --------------------------------------------------

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME,
    do_reduce_labels=False,
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = TorWICDataset(
    DATASET_ROOT,
    "train",
    processor,
)

print("Training images:", len(train_dataset))


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model.to(device)
model.train()


# --------------------------------------------------
# One training batch
# --------------------------------------------------

batch = next(iter(loader))

pixel_values = batch["pixel_values"].to(device)
labels = batch["labels"].to(device)

print()
print("Pixel tensor shape:", pixel_values.shape)
print("Label tensor shape:", labels.shape)
print("Unique labels:", torch.unique(labels).tolist())

outputs = model(
    pixel_values=pixel_values,
    labels=labels,
)

print()
print("Logits shape:", outputs.logits.shape)
print("Loss:", outputs.loss.item())


# --------------------------------------------------
# Backpropagation
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=6e-5,
    weight_decay=0.01,
)

optimizer.zero_grad()

outputs.loss.backward()

optimizer.step()

print()
print("One optimization step completed successfully.")
print("The model weights were updated.")

