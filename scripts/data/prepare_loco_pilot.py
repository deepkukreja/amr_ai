"""Prepare a bounded YOLO warehouse pilot from TUM's LOCO v1 release."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


CLASSES = {5: "forklift", 11: "pallet_truck", 7: "pallet", 3: "small_load_carrier", 10: "stillage"}
TRAIN_CAP = 68
EQUIPMENT_IDS = {5, 11}


def capture_split(member):
    path = PurePosixPath(member)
    if len(path.parts) < 3 or path.parts[0] != "dataset" or ".." in path.parts:
        raise ValueError(f"Unexpected LOCO archive member: {member}")
    _, site, capture, *_ = path.parts
    if site in {"subset-1", "subset-4"}:
        return "site" + site[-1]
    if site not in {"subset-2", "subset-3", "subset-5"}:
        raise ValueError(f"Unexpected LOCO site: {site}")
    return "val" if (site, capture) == ("subset-3", "2019-12-17_9") else "train"


def yolo_box(box, width, height):
    x, y, w, h = box
    if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
        raise ValueError(f"Invalid box {box} in a {width} x {height} image")
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def sample_training(members, objects, include_equipment_positives):
    count = len(members)
    if count <= TRAIN_CAP:
        return members
    selected = {(2 * i + 1) * count // (2 * TRAIN_CAP) for i in range(TRAIN_CAP)}
    if include_equipment_positives:
        for i, (_, image) in enumerate(members):
            if any(box["category_id"] in EQUIPMENT_IDS for box in objects.get(image["id"], ())):
                selected.add(i)
    return [members[i] for i in sorted(selected)]


def prepare(source, output, include_equipment_positives=False):
    labels = json.loads((source / "loco-all-v1.json").read_bytes())
    if {c["id"]: c["name"] for c in labels["categories"]} != CLASSES:
        raise ValueError("Unexpected LOCO class mapping")
    image_ids = {i["id"] for i in labels["images"]}
    if len(image_ids) != len(labels["images"]):
        raise ValueError("Duplicate LOCO image IDs")
    objects = defaultdict(list)
    for obj in labels["annotations"]:
        if obj["image_id"] not in image_ids or obj["category_id"] not in CLASSES or obj["iscrowd"]:
            raise ValueError("Unexpected image or category in LOCO annotation")
        objects[obj["image_id"]].append(obj)

    groups = defaultdict(list)
    for image in labels["images"]:
        member = image["path"].lstrip("/")
        split = capture_split(member)
        capture = "/".join(PurePosixPath(member).parts[:3]) if split == "train" else split
        groups[split, capture].append((member, image))

    output.mkdir(parents=True)
    rows = []
    digests = {}
    class_ids = {key: i for i, key in enumerate(CLASSES)}
    with ZipFile(source / "dataset.zip") as archive:
        for (split, _), members in sorted(groups.items()):
            members.sort()
            if split == "train":
                members = sample_training(members, objects, include_equipment_positives)
            images_dir = output / "images" / split
            labels_dir = output / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for member, image in members:
                pixels = archive.read(member)
                digest = hashlib.sha256(pixels).hexdigest()
                if digests.setdefault(digest, split) != split:
                    raise ValueError("An identical image crosses LOCO split boundaries")
                annotations = []
                for obj in objects[image["id"]]:
                    index = class_ids[obj["category_id"]]
                    box = yolo_box(obj["bbox"], image["width"], image["height"])
                    annotations.append(" ".join([str(index), *(f"{v:.9f}" for v in box)]))
                filename = f'{image["id"]:d}'
                (images_dir / (filename + ".jpg")).write_bytes(pixels)
                text = "".join(line + "\n" for line in annotations)
                (labels_dir / (filename + ".txt")).write_text(text)
                rows.append({"image_id": image["id"], "member": member, "split": split, "sha256": digest,
                             "label_sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "boxes": len(annotations)})

    config = {"path": str(output), "train": "images/train", "val": "images/val",
              "names": list(CLASSES.values())}
    for partition in ("site1", "site4"):
        (output / f"{partition}.yaml").write_text(json.dumps(
            {**config, "test": f"images/{partition}"}, indent=2) + "\n")
    (output / "data.yaml").write_text(json.dumps(config, indent=2) + "\n")
    manifest = json.dumps(rows, sort_keys=True, indent=2) + "\n"
    (output / "manifest.json").write_text(manifest)
    print(json.dumps({"frames": Counter(row["split"] for row in rows),
                      "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-equipment-positives", action="store_true",
                        help="Also keep all forklift and pallet-truck images in training captures")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.include_equipment_positives)


if __name__ == "__main__":
    main()
