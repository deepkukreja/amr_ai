import csv
import json
import os
import tkinter as tk

from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets/indian_manual_train_v1"
MANIFEST = DATASET / "manifest.csv"
ANNOTATIONS = DATASET / "manual_annotations"
PROGRESS = DATASET / "manual_annotation_progress.json"

CLASSES = {
    0: {"name": "person", "color": "#00c853"},
    1: {"name": "rider", "color": "#ff9800"},
}


def atomic_json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def read_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manual-evaluation manifest not found:\n{MANIFEST}"
        )

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise RuntimeError("The manual-evaluation manifest is empty.")

    if "relative_image_path" not in fields:
        raise RuntimeError(
            "Manifest is missing relative_image_path."
        )

    return fields, rows


class ManualAnnotator:
    def __init__(self, root):
        self.root = root
        self.fields, self.rows = read_manifest()

        self.total = len(self.rows)
        self.index = 0
        self.active_class = 0
        self.boxes = []
        self.selected_box = None
        self.dirty = False
        self.reviewed = False

        self.original_image = None
        self.tk_image = None
        self.image_width = 0
        self.image_height = 0

        self.image_x0 = 0
        self.image_y0 = 0
        self.display_width = 1
        self.display_height = 1

        self.drag_start = None

        ANNOTATIONS.mkdir(parents=True, exist_ok=True)

        # Resume at the first image that has not been reviewed.
        for i, row in enumerate(self.rows):
            if not self.is_reviewed(row):
                self.index = i
                break
        else:
            self.index = 0

        self.build_interface()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        self.root.bind_all("<Control-s>", self.save_shortcut)
        self.root.bind_all("<KeyPress-p>", self.choose_person)
        self.root.bind_all("<KeyPress-r>", self.choose_rider)
        self.root.bind_all("<KeyPress-n>", self.next_shortcut)
        self.root.bind_all("<KeyPress-b>", self.previous_shortcut)
        self.root.bind_all("<KeyPress-e>", self.empty_shortcut)
        self.root.bind_all("<Delete>", self.delete_shortcut)
        self.root.bind_all("<BackSpace>", self.delete_shortcut)
        self.root.bind_all("<Escape>", self.cancel_drawing)

        # Wait for the window to have a usable size.
        self.root.after(150, self.load_current)

    def annotation_path(self, row):
        stem = Path(row["relative_image_path"]).stem
        return ANNOTATIONS / f"{stem}.json"

    def is_reviewed(self, row):
        path = self.annotation_path(row)

        if not path.is_file():
            return False

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)

            return data.get("reviewed") is True

        except (OSError, json.JSONDecodeError):
            return False

    def reviewed_count(self):
        return sum(self.is_reviewed(row) for row in self.rows)

    def build_interface(self):
        self.root.title("AMR Manual Annotation Tool")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 650)

        header = ttk.Frame(self.root, padding=(10, 7))
        header.pack(fill="x")

        self.title_var = tk.StringVar()
        ttk.Label(
            header,
            textvariable=self.title_var,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")

        self.info_var = tk.StringVar()
        ttk.Label(
            header,
            textvariable=self.info_var,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(3, 4))

        instructions = (
            "DRAW: left-click and drag a box around a person. "
            "P = Person; R = Rider. Click a box to select it, "
            "then Delete to remove it. Ctrl+S saves."
        )

        ttk.Label(
            header,
            text=instructions,
            wraplength=1350,
        ).pack(anchor="w", pady=(0, 6))

        class_frame = ttk.Frame(header)
        class_frame.pack(fill="x")

        self.person_button = ttk.Button(
            class_frame,
            text="P — Person",
            command=lambda: self.set_class(0),
        )
        self.person_button.pack(side="left", padx=(0, 6))

        self.rider_button = ttk.Button(
            class_frame,
            text="R — Rider",
            command=lambda: self.set_class(1),
        )
        self.rider_button.pack(side="left")

        self.class_status = ttk.Label(class_frame, text="")
        self.class_status.pack(side="left", padx=12)

        ttk.Label(
            self.root,
            text=(
                "Green = Person | Orange = Rider | "
                "Cyan outline = Selected box"
            ),
        ).pack(anchor="w", padx=10, pady=(0, 4))

        self.canvas = tk.Canvas(
            self.root,
            background="#202124",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Configure>", self.on_resize)

        buttons = ttk.Frame(self.root, padding=(10, 7))
        buttons.pack(fill="x")

        ttk.Button(
            buttons,
            text="< Previous (B)",
            command=self.previous_image,
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            buttons,
            text="Next (N) >",
            command=self.next_image,
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons,
            text="Undo Last Box",
            command=self.undo_box,
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons,
            text="Delete Selected",
            command=self.delete_selected,
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons,
            text="Mark Empty (E)",
            command=self.mark_empty,
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons,
            text="Save / Reviewed (Ctrl+S)",
            command=self.save_current,
        ).pack(side="right", padx=(5, 0))

    def set_class(self, class_id):
        self.active_class = class_id

        name = CLASSES[class_id]["name"].upper()

        self.class_status.configure(
            text=f"Active drawing class: {name}"
        )

    def load_current(self):
        row = self.rows[self.index]
        image_path = DATASET / row["relative_image_path"]

        if not image_path.is_file():
            messagebox.showerror(
                "Missing image",
                f"Image not found:\n{image_path}",
                parent=self.root,
            )
            return

        with Image.open(image_path) as image:
            self.original_image = image.convert("RGB")

        self.image_width, self.image_height = (
            self.original_image.size
        )

        self.boxes = []
        self.selected_box = None
        self.dirty = False
        self.reviewed = False

        annotation_path = self.annotation_path(row)

        if annotation_path.is_file():
            try:
                with annotation_path.open(encoding="utf-8") as f:
                    annotation = json.load(f)

                self.boxes = annotation.get("boxes", [])
                self.reviewed = annotation.get("reviewed") is True

            except (OSError, json.JSONDecodeError) as exc:
                messagebox.showwarning(
                    "Could not load annotation",
                    f"{annotation_path}\n\n{exc}",
                    parent=self.root,
                )

        self.set_class(self.active_class)
        self.update_status()
        self.redraw()

    def update_status(self):
        row = self.rows[self.index]

        self.title_var.set(
            f"Image {self.index + 1}/{self.total} — "
            f"{Path(row['relative_image_path']).name}"
        )

        status = (
            "REVIEWED"
            if self.reviewed
            else "NOT REVIEWED"
        )

        dirty = " — UNSAVED CHANGES" if self.dirty else ""

        self.info_var.set(
            f"Clip: {row.get('clip_id', 'unknown')} | "
            f"Weather: {row.get('weather', 'unknown')} | "
            f"Time: {row.get('timeofday', 'unknown')} | "
            f"Scene: {row.get('scene', 'unknown')} | "
            f"Boxes: {len(self.boxes)} | "
            f"Reviewed: {self.reviewed_count()}/{self.total} | "
            f"{status}{dirty}"
        )

    def on_resize(self, _event=None):
        if self.original_image is not None:
            self.redraw()

    def redraw(self):
        if self.original_image is None:
            return

        canvas_width = max(self.canvas.winfo_width(), 200)
        canvas_height = max(self.canvas.winfo_height(), 150)

        max_width = max(100, canvas_width - 24)
        max_height = max(100, canvas_height - 24)

        scale = min(
            max_width / self.image_width,
            max_height / self.image_height,
        )

        self.display_width = max(
            1, round(self.image_width * scale)
        )
        self.display_height = max(
            1, round(self.image_height * scale)
        )

        self.image_x0 = (canvas_width - self.display_width) // 2
        self.image_y0 = (canvas_height - self.display_height) // 2

        resized = self.original_image.resize(
            (self.display_width, self.display_height),
            Image.Resampling.LANCZOS,
        )

        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")

        self.canvas.create_image(
            self.image_x0,
            self.image_y0,
            anchor="nw",
            image=self.tk_image,
        )

        for i, box in enumerate(self.boxes):
            class_id = int(box["class_id"])
            class_info = CLASSES[class_id]

            x1, y1, x2, y2 = self.normalized_to_canvas(
                box["bbox_norm_xyxy"]
            )

            color = class_info["color"]
            width = 3

            if i == self.selected_box:
                color = "#00e5ff"
                width = 4

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=width,
            )

            label = f"{class_info['name']} #{i + 1}"

            self.canvas.create_text(
                x1 + 3,
                y1 + 2,
                text=label,
                fill=color,
                anchor="nw",
                font=("Segoe UI", 9, "bold"),
            )

    def normalized_to_canvas(self, bbox):
        x1, y1, x2, y2 = bbox

        return (
            self.image_x0 + x1 * self.display_width,
            self.image_y0 + y1 * self.display_height,
            self.image_x0 + x2 * self.display_width,
            self.image_y0 + y2 * self.display_height,
        )

    def canvas_to_normalized(self, x, y):
        nx = (x - self.image_x0) / self.display_width
        ny = (y - self.image_y0) / self.display_height

        return (
            min(1.0, max(0.0, nx)),
            min(1.0, max(0.0, ny)),
        )

    def clamp_to_image(self, x, y):
        x = min(
            self.image_x0 + self.display_width,
            max(self.image_x0, x),
        )
        y = min(
            self.image_y0 + self.display_height,
            max(self.image_y0, y),
        )

        return x, y

    def mouse_down(self, event):
        x, y = self.clamp_to_image(event.x, event.y)

        # Clicking inside an existing box selects it instead of drawing.
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.normalized_to_canvas(
                self.boxes[i]["bbox_norm_xyxy"]
            )

            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_box = i
                self.drag_start = None
                self.redraw()
                return

        self.selected_box = None
        self.drag_start = (x, y)
        self.canvas.delete("preview")

    def mouse_drag(self, event):
        if self.drag_start is None:
            return

        x, y = self.clamp_to_image(event.x, event.y)
        start_x, start_y = self.drag_start

        color = CLASSES[self.active_class]["color"]

        self.canvas.delete("preview")

        self.canvas.create_rectangle(
            start_x, start_y, x, y,
            outline=color,
            width=3,
            dash=(5, 3),
            tags="preview",
        )

    def mouse_up(self, event):
        if self.drag_start is None:
            return

        start_x, start_y = self.drag_start
        end_x, end_y = self.clamp_to_image(event.x, event.y)

        self.drag_start = None
        self.canvas.delete("preview")

        if abs(end_x - start_x) < 3 or abs(end_y - start_y) < 3:
            self.redraw()
            return

        x1, y1 = self.canvas_to_normalized(start_x, start_y)
        x2, y2 = self.canvas_to_normalized(end_x, end_y)

        bbox = [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            self.redraw()
            return

        class_id = self.active_class

        self.boxes.append({
            "class_id": class_id,
            "class_name": CLASSES[class_id]["name"],
            "bbox_norm_xyxy": bbox,
        })

        self.selected_box = len(self.boxes) - 1
        self.dirty = True
        self.reviewed = False

        self.update_status()
        self.redraw()

    def undo_box(self):
        if not self.boxes:
            return

        self.boxes.pop()
        self.selected_box = None
        self.dirty = True
        self.reviewed = False

        self.update_status()
        self.redraw()

    def delete_selected(self):
        if self.selected_box is None:
            messagebox.showinfo(
                "Delete box",
                "Click inside a box to select it first.",
                parent=self.root,
            )
            return

        del self.boxes[self.selected_box]

        self.selected_box = None
        self.dirty = True
        self.reviewed = False

        self.update_status()
        self.redraw()

    def save_current(self, force_empty=False):
        if force_empty and self.boxes:
            confirm = messagebox.askyesno(
                "Mark empty",
                "This will delete ALL boxes on this image and mark it "
                "as intentionally empty.\n\nContinue?",
                parent=self.root,
            )

            if not confirm:
                return False

            self.boxes = []
            self.selected_box = None
            self.dirty = True

        elif not self.boxes and not force_empty:
            confirm = messagebox.askyesno(
                "Confirm empty image",
                "There are no boxes on this image.\n\n"
                "Have you checked the entire image and confirmed that "
                "it contains no people or riders?",
                parent=self.root,
            )

            if not confirm:
                return False

        row = self.rows[self.index]
        annotation_path = self.annotation_path(row)

        payload = {
            "image_filename": Path(
                row["relative_image_path"]
            ).name,
            "source_frame": row.get("frame_name", ""),
            "clip_id": row.get("clip_id", ""),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "reviewed": True,
            "empty": len(self.boxes) == 0,
            "classes": {
                str(class_id): info["name"]
                for class_id, info in CLASSES.items()
            },
            "boxes": self.boxes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        atomic_json_write(annotation_path, payload)

        self.reviewed = True
        self.dirty = False

        self.update_manifest_status()
        self.update_progress()
        self.update_status()
        self.redraw()

        print(
            f"Saved annotation: {annotation_path.name} | "
            f"boxes={len(self.boxes)} | "
            f"reviewed={self.reviewed_count()}/{self.total}",
            flush=True,
        )

        return True

    def update_manifest_status(self):
        # Update only the manual-evaluation manifest, not any training
        # dataset or the protected test dataset.
        if "manual_annotation_status" not in self.fields:
            self.fields.append("manual_annotation_status")

        for row in self.rows:
            row["manual_annotation_status"] = (
                "reviewed" if self.is_reviewed(row) else "pending"
            )

        temporary = MANIFEST.with_suffix(".csv.tmp")

        with temporary.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(self.rows)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary, MANIFEST)

    def update_progress(self):
        reviewed_files = [
            self.annotation_path(row).name
            for row in self.rows
            if self.is_reviewed(row)
        ]

        atomic_json_write(
            PROGRESS,
            {
                "total_images": self.total,
                "reviewed_images": len(reviewed_files),
                "pending_images": self.total - len(reviewed_files),
                "reviewed_annotation_files": reviewed_files,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def change_image(self, step):
        if self.dirty:
            choice = messagebox.askyesnocancel(
                "Unsaved changes",
                "Save your changes before changing images?\n\n"
                "Yes = Save\nNo = Discard changes\nCancel = Stay here",
                parent=self.root,
            )

            if choice is None:
                return

            if choice:
                if not self.save_current():
                    return

        self.index = (self.index + step) % self.total
        self.load_current()

    def next_image(self):
        self.change_image(1)

    def previous_image(self):
        self.change_image(-1)

    def mark_empty(self):
        if self.save_current(force_empty=True):
            # After confirming an empty image, proceed to the next one.
            self.index = (self.index + 1) % self.total
            self.load_current()

    def save_shortcut(self, _event=None):
        self.save_current()
        return "break"

    def choose_person(self, _event=None):
        self.set_class(0)

    def choose_rider(self, _event=None):
        self.set_class(1)

    def next_shortcut(self, _event=None):
        self.next_image()

    def previous_shortcut(self, _event=None):
        self.previous_image()

    def empty_shortcut(self, _event=None):
        self.mark_empty()

    def delete_shortcut(self, _event=None):
        self.delete_selected()
        return "break"

    def cancel_drawing(self, _event=None):
        self.drag_start = None
        self.canvas.delete("preview")
        self.redraw()

    def close_app(self):
        if self.dirty:
            choice = messagebox.askyesnocancel(
                "Unsaved changes",
                "Save your changes before closing?",
                parent=self.root,
            )

            if choice is None:
                return

            if choice and not self.save_current():
                return

        self.root.destroy()


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(
            f"Manual dataset not found:\n{DATASET}"
        )

    root = tk.Tk()
    app = ManualAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
