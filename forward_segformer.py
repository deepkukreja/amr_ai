from pathlib import Path

import torch
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
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

LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}

# Automatically select the first test image
test_dir = Path(
    "datasets/torwic_segmentation/split/test/images"
)

image_files = sorted(test_dir.glob("*.png"))

if not image_files:
    raise FileNotFoundError(
        f"No test images found in {test_dir}"
    )

image_path = image_files[0]

mask_path = Path(
    "datasets/torwic_segmentation/split/test/masks"
) / image_path.name

if not mask_path.exists():
    raise FileNotFoundError(
        f"Matching mask not found: {mask_path}"
    )

print("Image:", image_path)
print("Mask :", mask_path)

# Load image
image = Image.open(image_path).convert("RGB")

print("Original image size:", image.size)

# Load processor
processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME
)

# Load pretrained SegFormer encoder + new 16-class head
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
model.eval()

print("Device:", device)

# Convert RGB image to model input tensor
inputs = processor(
    images=image,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("Input tensor shape:")
print(inputs["pixel_values"].shape)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

print("Raw logits shape:")
print(outputs.logits.shape)

# Choose the most likely class at every output pixel
prediction = outputs.logits.argmax(dim=1)

print("Prediction tensor shape:")
print(prediction.shape)

unique_classes = torch.unique(
    prediction
).cpu().numpy()

print()
print("Predicted class IDs:")

for class_id in unique_classes:
    class_id = int(class_id)
    print(
        f"{class_id:>2} -> "
        f"{ID2LABEL[class_id]}"
    )

print()
print("Forward pass successful.")
