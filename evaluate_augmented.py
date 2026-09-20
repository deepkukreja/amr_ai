from pathlib import Path

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

DATASET_ROOT = Path("datasets/torwic_segmentation/split")
MODEL_DIR = Path("runs/segformer_torwic_augmented/best")

NUM_CLASSES = 16
BATCH_SIZE = 1

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
# DATASET
# ============================================================

class TorWICDataset(Dataset):

    def __init__(self, root, split, processor):
        self.root = Path(root)
        self.split = split
        self.processor = processor

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        self.images = sorted(self.image_dir.glob("*.png"))
        self.masks = [
            self.mask_dir / image_path.name
            for image_path in self.images
        ]

        print(f"{split.capitalize()} images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load RGB image
        image = Image.open(image_path).convert("RGB")

        # Load segmentation mask
        mask = Image.open(mask_path).convert("L")

        # Process image for SegFormer
        encoded = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = encoded["pixel_values"].squeeze(0)

        # ----------------------------------------------------
        # IMPORTANT:
        # Resize mask to exactly the model input resolution.
        # Use NEAREST so class IDs are not blended.
        # ----------------------------------------------------

        target_height = pixel_values.shape[-2]
        target_width = pixel_values.shape[-1]

        mask = mask.resize(
            (target_width, target_height),
            Image.Resampling.NEAREST
        )

        labels = torch.tensor(
            np.array(mask, dtype=np.int64),
            dtype=torch.long
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# PROCESSOR
# ============================================================

processor = SegformerImageProcessor.from_pretrained(
    MODEL_DIR
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = TorWICDataset(
    DATASET_ROOT,
    "test",
    processor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR
)

model = model.to(device)
model.eval()


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

total_correct = 0
total_pixels = 0


# ============================================================
# EVALUATION
# ============================================================

print("Evaluating test set...")

with torch.no_grad():

    for batch_idx, batch in enumerate(test_loader):

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            pixel_values=pixel_values
        )

        # SegFormer outputs lower-resolution logits
        # Resize them to label resolution
        logits = F.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # Pixel accuracy
        # ----------------------------------------------------

        total_correct += (
            predictions == labels
        ).sum().item()

        total_pixels += labels.numel()

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        preds_np = predictions.cpu().numpy().reshape(-1)
        labels_np = labels.cpu().numpy().reshape(-1)

        for true_class, predicted_class in zip(
            labels_np,
            preds_np
        ):

            if (
                0 <= true_class < NUM_CLASSES
                and 0 <= predicted_class < NUM_CLASSES
            ):
                confusion[
                    true_class,
                    predicted_class
                ] += 1

        print(
            f"Processed batch "
            f"{batch_idx + 1}/{len(test_loader)}"
        )


# ============================================================
# CALCULATE IoU
# ============================================================

ious = []

print()
print("=" * 72)
print("PER-CLASS IoU")
print("=" * 72)

for class_id in range(NUM_CLASSES):

    true_positive = confusion[
        class_id,
        class_id
    ]

    false_positive = (
        confusion[:, class_id].sum()
        - true_positive
    )

    false_negative = (
        confusion[class_id, :].sum()
        - true_positive
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union == 0:

        iou = np.nan
        iou_text = "N/A"

    else:

        iou = true_positive / union
        iou_text = f"{iou:.4f}"

    ious.append(iou)

    print(
        f"{class_id:>2}  "
        f"{ID2LABEL[class_id]:<28}  "
        f"IoU: {iou_text}"
    )


# ============================================================
# OVERALL METRICS
# ============================================================

mIoU = np.nanmean(ious)

pixel_accuracy = (
    total_correct / total_pixels
)


print()
print("-" * 72)

print(
    f"Pixel Accuracy: {pixel_accuracy:.4f}"
)

print(
    f"Mean IoU:       {mIoU:.4f}"
)

print("-" * 72)
