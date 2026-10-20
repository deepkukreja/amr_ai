from pathlib import Path
import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]

SEW_ROOT = PROJECT / "datasets" / "sew"
V6_ROOT = PROJECT / "datasets" / "torwic_sew_adapt_v6_real"
V7_ROOT = PROJECT / "datasets" / "torwic_sew_v7_lowlight"

SESSION_FILE = SEW_ROOT / "v3_train_sessions.txt"


def lowlight_transform(img, rng):
    # Work on luminance so color information is largely preserved.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Random exposure reduction.
    exposure = rng.uniform(0.32, 0.50)
    l = np.clip(
        l.astype(np.float32) * exposure,
        0,
        255,
    ).astype(np.uint8)

    # Restore local contrast after exposure reduction.
    clip_limit = rng.uniform(1.4, 2.2)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(8, 8),
    )
    l = clahe.apply(l)

    # Match the joint brightness/contrast region of the real dark domain
    # without forcing every image to exactly the same statistics.
    target_mean = rng.uniform(38.0, 55.0)
    target_std = rng.uniform(32.0, 48.0)

    current_mean = float(l.mean())
    current_std = float(l.std())

    if current_std > 1e-6:
        adjusted = (
            (l.astype(np.float32) - current_mean)
            * (target_std / current_std)
            + target_mean
        )

        # Keep some natural variation from the CLAHE image.
        blend = rng.uniform(0.65, 0.90)
        l = (
            blend * adjusted
            + (1.0 - blend) * l.astype(np.float32)
        )

    l = np.clip(l, 0, 255).astype(np.uint8)

    # Slight low-light color desaturation.
    result = cv2.cvtColor(
        cv2.merge([l, a, b]),
        cv2.COLOR_LAB2BGR,
    )

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    saturation_scale = rng.uniform(0.85, 1.00)
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * saturation_scale,
        0,
        255,
    ).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Mild sensor noise.
    sigma = rng.uniform(1.0, 3.0)
    noise = rng.normal(
        0.0,
        sigma,
        result.shape,
    ).astype(np.float32)

    result = np.clip(
        result.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    return result


def main():
    if not V6_ROOT.exists():
        raise RuntimeError(
            f"V6 dataset not found: {V6_ROOT}"
        )

    sessions = [
        Path(x.strip())
        for x in SESSION_FILE.read_text().splitlines()
        if x.strip()
    ]

    # Only real SEW indoor training sessions.
    indoor_sessions = [
        s for s in sessions
        if s.name.lower().startswith("indoor__")
    ]

    source_images = []

    for session in indoor_sessions:
        source_images.extend(
            sorted((session / "images").glob("*.jpg"))
        )

    # Remove duplicates while preserving deterministic ordering.
    source_images = sorted(
        set(source_images),
        key=lambda p: str(p),
    )

    print(f"Indoor training sessions: {len(indoor_sessions)}")
    print(f"Indoor source images:     {len(source_images)}")

    if len(source_images) == 0:
        raise RuntimeError("No indoor SEW training images found.")

    out_images = V7_ROOT / "images" / "train"
    out_labels = V7_ROOT / "labels" / "train"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260929)

    generated = 0
    missing_labels = 0

    # One variant from every indoor source image.
    jobs = [(p, 0) for p in source_images]

    # Second variant for half of the source images.
    jobs.extend(
        (p, 1)
        for i, p in enumerate(source_images)
        if i % 2 == 0
    )

    target_count = 450
    jobs = jobs[:target_count]

    for source, variant in jobs:
        label = source.parent.parent / "labels" / f"{source.stem}.txt"

        if not label.exists():
            missing_labels += 1
            continue

        img = cv2.imread(str(source))

        if img is None:
            continue

        result = lowlight_transform(img, rng)

        name = (
            f"v7_lowlight_{source.parent.parent.name}_"
            f"{source.stem}_v{variant}.jpg"
        )

        out_img = out_images / name
        out_label = out_labels / f"{Path(name).stem}.txt"

        cv2.imwrite(
            str(out_img),
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                int(rng.integers(88, 96)),
            ],
        )

        out_label.write_text(
            label.read_text()
        )

        generated += 1

    print(f"Generated images:        {generated}")
    print(f"Missing source labels:   {missing_labels}")

    if generated == 0:
        raise RuntimeError("No V7 images were generated.")

    # Create combined training manifest:
    # all V6 training images + V7 low-light images.
    manifest = V7_ROOT / "v7_train_images.txt"

    lines = []

    v6_images = sorted(
        (V6_ROOT / "images" / "train").glob("*")
    )

    for p in v6_images:
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lines.append(str(p.resolve()))
    v7_images = sorted(
        out_images.glob("*.jpg")
    )

    for p in v7_images:
        lines.append(str(p.resolve()))

    manifest.write_text(
        "\n".join(lines) + "\n"
    )

    yaml = V7_ROOT / "torwic_sew_v7_lowlight.yaml"

    yaml.write_text(
        f"""path: {V7_ROOT}
train: v7_train_images.txt
val: ../torwic_sew_adapt_v6_real/images/val
test: ../torwic_sew_adapt_v6_real/images/val
names:
  0: person
  1: fork_truck
  2: cart_pallet_jack
  3: rack_shelf
  4: misc_dynamic_feature
"""
    )

    print(f"Training manifest:       {manifest}")
    print(f"Dataset YAML:            {yaml}")

    # Audit generated low-light images.
    brightness = []
    contrast = []

    for p in v7_images:
        im = cv2.imread(
            str(p),
            cv2.IMREAD_GRAYSCALE,
        )
        if im is not None:
            brightness.append(float(im.mean()))
            contrast.append(float(im.std()))

    print("\nV7 LOW-LIGHT AUDIT")
    print("-" * 50)
    print(f"Images:             {len(brightness)}")
    print(f"Mean brightness:    {np.mean(brightness):.1f}")
    print(f"Median brightness:  {np.median(brightness):.1f}")
    print(f"Mean contrast:      {np.mean(contrast):.1f}")
    print(f"Median contrast:    {np.median(contrast):.1f}")


if __name__ == "__main__":
    main()
