from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("datasets/torwic_segmentation/split")
MODEL_DIR = Path("runs/segformer_torwic_weighted/best")

OUTPUT_DIR = Path(
    "runs/segformer_torwic_weighted/visualizations"
)

NUM_IMAGES = 10


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
# COLORS FOR VISUALIZATION
# ============================================================

COLORS = np.array(
    [
        [0, 0, 0],          # 0 background
        [128, 128, 128],    # 1 driveable_ground
        [180, 220, 255],    # 2 ceiling
        [255, 255, 255],    # 3 ego_vehicle
        [150, 75, 0],       # 4 wall
        [100, 100, 100],    # 5 misc static
        [0, 120, 255],      # 6 rack shelf
        [255, 180, 0],      # 7 goods
        [180, 0, 180],      # 8 machinery
        [255, 0, 0],        # 9 cart/pallet jack
        [255, 255, 0],      # 10 pylon
        [0, 255, 255],      # 11 text
        [255, 0, 255],      # 12 misc non-static
        [0, 255, 0],        # 13 person
        [255, 80, 80],      # 14 fork truck
        [0, 150, 150],      # 15 dynamic
    ],
    dtype=np.uint8,
)


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
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("Loading processor...")

processor = SegformerImageProcessor.from_pretrained(
    MODEL_DIR
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR
)

model = model.to(device)
model.eval()


# ============================================================
# TEST IMAGES
# ============================================================

image_dir = DATASET_ROOT / "test" / "images"
mask_dir = DATASET_ROOT / "test" / "masks"

image_paths = sorted(
    image_dir.glob("*.png")
)

image_paths = image_paths[:NUM_IMAGES]

print(
    f"Visualizing {len(image_paths)} test images..."
)


# ============================================================
# FUNCTION: COLORIZE MASK
# ============================================================

def colorize_mask(mask):

    mask_array = np.array(
        mask,
        dtype=np.int64
    )

    color_mask = np.zeros(
        (
            mask_array.shape[0],
            mask_array.shape[1],
            3
        ),
        dtype=np.uint8
    )

    for class_id in range(len(COLORS)):

        color_mask[
            mask_array == class_id
        ] = COLORS[class_id]

    return Image.fromarray(
        color_mask
    )


# ============================================================
# FUNCTION: ADD TITLE
# ============================================================

def add_title(image, title):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [0, 0, image.width, 40],
        fill=(0, 0, 0)
    )

    draw.text(
        (10, 10),
        title,
        fill=(255, 255, 255)
    )

    return image


# ============================================================
# PROCESS IMAGES
# ============================================================

with torch.no_grad():

    for index, image_path in enumerate(image_paths):

        print(
            f"Processing {index + 1}/"
            f"{len(image_paths)}: "
            f"{image_path.name}"
        )

        mask_path = (
            mask_dir / image_path.name
        )

        # ----------------------------------------------------
        # Load original image and ground truth
        # ----------------------------------------------------

        original = Image.open(
            image_path
        ).convert("RGB")

        ground_truth = Image.open(
            mask_path
        ).convert("L")

        original_width, original_height = (
            original.size
        )

        # ----------------------------------------------------
        # Prepare image for model
        # ----------------------------------------------------

        encoded = processor(
            images=original,
            return_tensors="pt"
        )

        pixel_values = (
            encoded["pixel_values"]
            .to(device)
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        outputs = model(
            pixel_values=pixel_values
        )

        # SegFormer produces lower-resolution logits
        logits = F.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        prediction = torch.argmax(
            logits,
            dim=1
        )

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # ----------------------------------------------------
        # Resize prediction back to ORIGINAL image size
        # ----------------------------------------------------

        prediction_image = Image.fromarray(
            prediction,
            mode="L"
        )

        prediction_image = prediction_image.resize(
            (
                original_width,
                original_height
            ),
            Image.Resampling.NEAREST
        )

        # ----------------------------------------------------
        # Convert masks to colors
        # ----------------------------------------------------

        gt_color = colorize_mask(
            ground_truth
        )

        prediction_color = colorize_mask(
            prediction_image
        )

        # ----------------------------------------------------
        # Resize everything to common display size
        # ----------------------------------------------------

        display_width = 640

        scale = (
            display_width /
            original_width
        )

        display_height = int(
            original_height * scale
        )

        original_display = original.resize(
            (
                display_width,
                display_height
            )
        )

        gt_color = gt_color.resize(
            (
                display_width,
                display_height
            ),
            Image.Resampling.NEAREST
        )

        prediction_color = prediction_color.resize(
            (
                display_width,
                display_height
            ),
            Image.Resampling.NEAREST
        )

        # ----------------------------------------------------
        # Add titles
        # ----------------------------------------------------

        original_display = add_title(
            original_display,
            "ORIGINAL"
        )

        gt_color = add_title(
            gt_color,
            "GROUND TRUTH"
        )

        prediction_color = add_title(
            prediction_color,
            "MODEL PREDICTION"
        )

        # ----------------------------------------------------
        # Create side-by-side comparison
        # ----------------------------------------------------

        panel_width = display_width
        panel_height = (
            display_height + 40
        )

        comparison = Image.new(
            "RGB",
            (
                panel_width * 3,
                panel_height
            ),
            (255, 255, 255)
        )

        comparison.paste(
            original_display,
            (0, 0)
        )

        comparison.paste(
            gt_color,
            (panel_width, 0)
        )

        comparison.paste(
            prediction_color,
            (panel_width * 2, 0)
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            f"comparison_{index + 1:02d}.png"
        )

        comparison.save(
            output_path
        )

        print(
            f"Saved: {output_path}"
        )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 72)
print("VISUALIZATION COMPLETE")
print("=" * 72)
print(
    f"Results saved to:\n{OUTPUT_DIR}"
)
print("=" * 72)
