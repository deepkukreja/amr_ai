import torch

from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

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

print("Loading SegFormer-B0...")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print()
print("Model loaded successfully")
print("Device:", device)
print("Number of classes:", model.config.num_labels)
print("Model parameters:", sum(p.numel() for p in model.parameters()))
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
