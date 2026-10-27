"""Visual attributes for each cat photo.

YOLO finds the cat and its mask. Fur colour and pattern are measured on those
pixels. Breed, framing, view, eyes, age, and posture are CLIP labels on the crop.

From the project root:

    python3.12 -m venv .venv
    .venv/bin/pip install -U -r requirements.txt
    .venv/bin/python photo-labelling/label_photos.py

Weights go in models/yolo11n-seg.pt. Download:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
The script fetches that file into models/ if it is not already there.

Writes prepared/photo_labels.csv, prepared/cat_labels.csv, and a
captioned copy of each photo under prepared/captioned-photos/.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("YOLO_VERBOSE", "False")
warnings.filterwarnings("ignore", module="ultralytics")
warnings.filterwarnings("ignore", module="torch")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "cat-arena" / "photos"
PROCESSED = ROOT / "prepared"
WEIGHTS = ROOT / "models" / "yolo11n-seg.pt"
LABEL_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

THREADS = 8
YOLO_IMGSZ = 512
YOLO_CONF = 0.05
YOLO_DEVICE = "cpu"
YOLO_BATCH = 8
CAT_CLASS_NAME = "cat"
CAT_CONF = 0.20
DOG_CONF = 0.45

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_QUICK_GELU = True
TEMPLATES = ("a photo of {}", "a picture of {}")

COLOURS = ["black", "white", "gray", "orange", "cream", "brown"]
BREEDS = [
    ("domestic_shorthair", "a domestic shorthair cat"),
    ("domestic_longhair", "a longhaired domestic cat with fluffy fur"),
    ("siamese", "a siamese cat with a pale body and dark face, ears and tail"),
    ("persian", "a persian cat with a flat face and long fur"),
    ("maine_coon", "a maine coon cat with tufted ears and a ruff"),
    ("bengal", "a bengal cat with a spotted or marbled coat"),
    ("sphynx", "a hairless sphynx cat"),
    ("british_shorthair", "a british shorthair cat with a round face and thick short fur"),
]
FRAMING = [
    ("close_up", "a close-up of a cat's face filling the frame"),
    ("medium", "a cat photographed from the chest up"),
    ("full_body", "a full body view of a cat"),
]
VIEWS = [
    ("front", "a cat facing the camera"),
    ("three_quarter", "a cat with its face turned slightly to the side"),
    ("profile", "a cat in side profile"),
    ("from_above", "a cat seen from above"),
    ("looking_away", "a cat looking away from the camera"),
    ("rear", "a cat seen from behind"),
]
EYES = [
    ("both_open", "a cat with both eyes open"),
    ("half_closed", "a cat with half-closed sleepy eyes"),
    ("closed", "a cat with its eyes closed"),
    ("not_visible", "a cat whose eyes cannot be seen"),
]
EYE_COLOURS = [
    ("green", "a cat with green eyes"),
    ("yellow", "a cat with yellow eyes"),
    ("blue", "a cat with blue eyes"),
    ("copper", "a cat with copper orange eyes"),
]
AGE = [
    ("kitten", "a young kitten with a small body and a large head"),
    ("adult", "a fully grown adult cat"),
]
POSTURE = [
    ("sitting", "a cat sitting upright"),
    ("lying", "a cat lying down"),
    ("standing", "a cat standing on all four paws"),
]
PROMPT_GROUPS = {
    "breed": BREEDS,
    "framing": FRAMING,
    "view": VIEWS,
    "eyes": EYES,
    "eyes_colour": EYE_COLOURS,
    "age": AGE,
    "posture": POSTURE,
}

# A specific breed is kept only when CLIP's margin is at least this large.
BREED_MARGIN = 0.42
# Box size overrides CLIP framing when the shot is obvious.
CLOSE_UP_AREA = 0.72
CLOSE_UP_EDGES = 3
CLOSE_UP_EDGE_AREA = 0.40
FULL_BODY_AREA = 0.22
FULL_BODY_EDGES = 1
CROP_PAD = 0.06

# A dark pupil with a catchlight overrides the CLIP eye label.
OPEN_EYE_SINGLE = 140
OPEN_EYE_PAIR = 100
OPEN_EYE_SCORE = 0.85
CLOSED_TO_OPEN_SCORE = 0.7
EYE_COLOUR_MIN = 0.35
EYE_COLOUR_BLOB = 0.60
FACE_VIEWS = {"front", "three_quarter"}
FACE_FRAMING = {"close_up", "medium"}
AWAY_VIEWS = {"rear", "looking_away"}

BOX_FIELDS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
DROP_COLUMNS = ("breed_raw", "breed_second", "eyes_colour_clip", "eyes_colour_clip_score")
LABEL_JPEG_QUALITY = 85


def main() -> None:
    from save_photo_labels import (
        CAT_INT,
        IMAGE_INT,
        aggregate,
        apply_open_eyes,
        export_labelled_images,
        fill_boxes,
        output_paths,
        run,
        tidy_frame,
    )
    import pandas as pd

    parser = argparse.ArgumentParser(description="Label breed, colour, camera angle, eyes, age, and posture for each cat.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N photos, in filename order.")
    parser.add_argument("--sample", type=int, default=0, help="One photo from this many evenly spaced cats.")
    parser.add_argument("--batch", type=int, default=YOLO_BATCH)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument("--tag", default="", help="Write photo_labels_<tag>.csv instead of the main files.")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing image CSV and start again.")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild the per-cat CSV from the image CSV.")
    parser.add_argument("--labels-only", action="store_true", help="Write captioned photos from the image CSV.")
    parser.add_argument("--boxes-only", action="store_true", help="Detect boxes, store them, and redraw the labelled photos.")
    parser.add_argument("--eyes-only", action="store_true", help="Correct open eyes from the pupil check and redraw the labelled photos.")
    args = parser.parse_args()
    if args.eyes_only:
        image_csv, cat_csv = output_paths(args.tag)
        images, changed = apply_open_eyes(pd.read_csv(image_csv))
        images = tidy_frame(images, IMAGE_INT)
        images.to_csv(image_csv, index=False, na_rep="")
        cats = tidy_frame(aggregate(images), CAT_INT)
        cats.to_csv(cat_csv, index=False, na_rep="")
        export_labelled_images(images)
        print(f"Corrected {changed} eye labels", flush=True)
        return
    if args.boxes_only:
        fill_boxes(args.batch, args.imgsz, args.tag)
        return
    if args.labels_only:
        image_csv, _ = output_paths(args.tag)
        export_labelled_images(pd.read_csv(image_csv))
        return
    if args.aggregate_only:
        image_csv, cat_csv = output_paths(args.tag)
        images = pd.read_csv(image_csv)
        cats = tidy_frame(aggregate(images), CAT_INT)
        cats.to_csv(cat_csv, index=False, na_rep="")
        export_labelled_images(images)
        print(f"Wrote {cat_csv} ({len(cats)} cats)", flush=True)
        return
    run(args.limit, args.sample, args.batch, args.imgsz, args.tag, args.fresh)


if __name__ == "__main__":
    main()
