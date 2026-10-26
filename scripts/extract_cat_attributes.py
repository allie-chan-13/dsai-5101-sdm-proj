"""Visual attributes for each cat photo.

YOLO (yolo11n-seg) finds the cat and its mask. Coat colour and pattern are
measured on those pixels. Breed, framing, view, eyes, age, and posture are
zero-shot CLIP labels on the cropped cat.

From the project root:

    python3.12 -m venv .venv
    .venv/bin/pip install -U -r requirements.txt
    .venv/bin/python scripts/extract_cat_attributes.py

Writes processed/image_attributes.csv (one row per photo) and
processed/cat_attributes.csv (one row per cat). Strength columns are joined
from processed/cat_strengths.csv when that file is present.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "dataset" / "images"
PROCESSED = ROOT / "processed"
WEIGHTS = ROOT / "models" / "yolo11n-seg.pt"
CAT_CLASS_NAME = "cat"

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
TEMPLATES = ("a photo of {}", "a picture of {}")


def list_images() -> list[Path]:
    paths = sorted(IMAGE_ROOT.glob("cat_*/*.jpg"))
    if not paths:
        raise SystemExit(f"No cat images under {IMAGE_ROOT}")
    return paths


def select_paths(paths: list[Path], limit: int, sample: int) -> list[Path]:
    if sample:
        by_cat: dict[str, list[Path]] = {}
        for path in paths:
            by_cat.setdefault(path.parent.name, []).append(path)
        cats = sorted(by_cat)
        if sample < len(cats):
            idx = np.linspace(0, len(cats) - 1, sample).round().astype(int)
            cats = list(dict.fromkeys(cats[i] for i in idx.tolist()))
        return [sorted(by_cat[cat])[0] for cat in cats]
    if limit:
        return paths[:limit]
    return paths


def load_yolo():
    from ultralytics import YOLO

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        prev = os.getcwd()
        os.chdir(WEIGHTS.parent)
        try:
            YOLO(WEIGHTS.name)
        finally:
            os.chdir(prev)
    model = YOLO(str(WEIGHTS))
    cat_ids = [i for i, name in model.names.items() if name == CAT_CLASS_NAME]
    if not cat_ids:
        raise SystemExit(f"{WEIGHTS.name} has no '{CAT_CLASS_NAME}' class: {model.names}")
    return model, cat_ids


def load_clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, preprocess, tokenizer


def encode_phrases(model, tokenizer, phrases: list[str]) -> torch.Tensor:
    encoded = []
    for phrase in phrases:
        tokens = tokenizer([template.format(phrase) for template in TEMPLATES])
        with torch.inference_mode():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            pooled = features.mean(dim=0)
            pooled = pooled / pooled.norm()
        encoded.append(pooled)
    return torch.stack(encoded)


def prepare_text(model, tokenizer) -> dict[str, tuple[list[str], torch.Tensor]]:
    prepared = {}
    for name, pairs in PROMPT_GROUPS.items():
        labels = [label for label, _ in pairs]
        phrases = [phrase for _, phrase in pairs]
        prepared[name] = (labels, encode_phrases(model, tokenizer, phrases))
    return prepared


def animal_instance(result):
    """Largest confident cat. Some upside-down cats are labelled dog; use that mask then."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    height, width = result.orig_shape
    candidates = []
    for i, class_id in enumerate(cls):
        name = result.names[int(class_id)]
        if name not in {"cat", "dog"}:
            continue
        x1, y1, x2, y2 = xyxy[i]
        area = max(0.0, float(x2 - x1) * float(y2 - y1))
        candidates.append((name, float(conf[i]), area, i))
    cats = [item for item in candidates if item[0] == "cat" and item[1] >= 0.20]
    dogs = [item for item in candidates if item[0] == "dog" and item[1] >= 0.45]
    pool = cats or dogs
    if not pool or height == 0 or width == 0:
        return None
    name, score, area, best = max(pool, key=lambda item: item[1] * np.sqrt(item[2]))
    mask = np.zeros((height, width), dtype=bool)
    if result.masks is not None and best < len(result.masks.xy):
        poly = result.masks.xy[best]
        if len(poly):
            drawn = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(drawn, [np.asarray(poly, dtype=np.int32)], 1)
            mask = drawn.astype(bool)
    if int(mask.sum()) < 50:
        x1, y1, x2, y2 = [int(v) for v in xyxy[best]]
        mask = np.zeros((height, width), dtype=bool)
        mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
        source = "bbox"
    else:
        source = "mask"
    return {
        "name": name,
        "conf": score,
        "xyxy": xyxy[best],
        "mask": mask,
        "source": source,
        "area_frac": area / float(height * width),
    }


def edges_touched(mask: np.ndarray, band: int = 2) -> int:
    if mask is None or mask.sum() == 0:
        return 0
    touches = 0
    if mask[:band, :].any():
        touches += 1
    if mask[-band:, :].any():
        touches += 1
    if mask[:, :band].any():
        touches += 1
    if mask[:, -band:].any():
        touches += 1
    return touches


def name_coat_pixels(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Map HSV pixels to coat names. Later assignments win, so black is last.

    White is only near-neutral and very bright, so pale gray tabby stripes stay gray.
    Warm indoor light still shifts hue, but low saturation keeps that fur gray.
    """
    names = np.full(hue.shape, "gray", dtype=object)
    names[(sat >= 50) & (hue > 40) & (hue < 150)] = "other"
    names[(hue <= 32) & (sat >= 70) & (val >= 40) & (val < 180)] = "brown"
    names[(hue <= 26) & (sat >= 100) & (val >= 85)] = "orange"
    names[(sat < 50) & (val >= 45) & (val < 205)] = "gray"
    # Pale warm fur is cream, including pixels the gray rule would otherwise take.
    names[(hue <= 32) & (sat >= 18) & (sat < 75) & (val >= 160)] = "cream"
    names[(sat <= 25) & (val >= 200)] = "white"
    names[(val < 50) | ((sat < 30) & (val < 75))] = "black"
    return names


def pointed_score(value: np.ndarray, mask: np.ndarray) -> float:
    ys = np.flatnonzero(mask.any(axis=1))
    if len(ys) < 2:
        return 0.0
    y0, y1 = int(ys[0]), int(ys[-1])
    span = max(1, y1 - y0)
    top = mask.copy()
    top[y0 + int(0.22 * span) :, :] = False
    body = mask.copy()
    body[: y0 + int(0.35 * span), :] = False
    body[y0 + int(0.80 * span) :, :] = False
    top_px = value[top]
    body_px = value[body]
    if top_px.size < 20 or body_px.size < 20:
        return 0.0
    return float(np.clip((body_px.mean() - top_px.mean()) / 255.0, 0, 1))


def tabby_score(value: np.ndarray, names: np.ndarray) -> float:
    coat = np.isin(names, ["gray", "brown", "orange"])
    vals = value[coat]
    if vals.size < 40:
        return 0.0
    spread = float(vals.std())
    median = float(np.median(vals))
    dark = float((vals < median).mean())
    balance = min(dark, 1.0 - dark) / 0.5
    return float(np.clip((spread - 18.0) / 28.0, 0, 1) * balance)


def labels_from_coat(shares: dict[str, float], tabby: float, pointed: float) -> tuple[str, str, str]:
    total = sum(shares.values())
    if total <= 0:
        return "unknown", "", "unknown"
    ordered = sorted(COLOURS, key=lambda name: shares[name], reverse=True)
    primary = ordered[0]
    secondary = ordered[1] if shares[ordered[1]] >= 0.12 else ""
    black, white, orange, cream, brown = (
        shares["black"],
        shares["white"],
        shares["orange"],
        shares["cream"],
        shares["brown"],
    )
    light = white + cream
    if black >= 0.07 and orange >= 0.06 and white >= 0.12 and (orange + brown) >= 0.12:
        pattern = "calico"
    elif orange >= 0.15 and black >= 0.12 and white < 0.12:
        pattern = "tortoiseshell"
    elif pointed >= 0.12 and light >= 0.45 and black < 0.25:
        pattern = "pointed"
    elif (black + brown) >= 0.65 and orange < 0.08 and light < 0.12:
        pattern = "solid"
        primary = "black" if black >= brown else "brown"
        secondary = ""
    elif tabby >= 0.45 and primary in {"gray", "brown", "orange"}:
        pattern = "tabby"
    elif shares[ordered[1]] >= 0.18:
        pattern = "bicolor"
    else:
        pattern = "solid"
    return primary, secondary, pattern


def coat_features(bgr: np.ndarray, mask: np.ndarray) -> dict:
    empty = {f"share_{name}": 0.0 for name in COLOURS}
    empty.update(
        colour_primary="unknown",
        colour_secondary="",
        colour_pattern="unknown",
        tabby_score=0.0,
        pointed_score=0.0,
    )
    if mask is None or int(mask.sum()) < 30:
        return empty
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][mask].astype(np.int16)
    sat = hsv[:, :, 1][mask].astype(np.int16)
    val = hsv[:, :, 2][mask].astype(np.int16)
    names = name_coat_pixels(hue, sat, val)
    keep = names != "other"
    if int(keep.sum()) < 30:
        keep = np.ones(len(names), dtype=bool)
    kept = names[keep]
    shares = {name: float((kept == name).mean()) if len(kept) else 0.0 for name in COLOURS}
    tabby = tabby_score(val[keep], kept)
    pointed = pointed_score(hsv[:, :, 2], mask)
    primary, secondary, pattern = labels_from_coat(shares, tabby, pointed)
    row = {f"share_{name}": shares[name] for name in COLOURS}
    row.update(
        colour_primary=primary,
        colour_secondary=secondary,
        colour_pattern=pattern,
        tabby_score=tabby,
        pointed_score=pointed,
    )
    return row


def eye_colour_from_mask(bgr: np.ndarray, mask: np.ndarray) -> str:
    """Colour of one or two small round iris-like blobs. Empty when none are found."""
    if mask is None or int(mask.sum()) < 50:
        return ""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = mask & (hue >= 95) & (hue <= 130) & (sat > 25) & (val > 90) & (val < 245)
    warm = mask & (val > 70) & (val < 245) & (sat > 60) & (hue >= 5) & (hue <= 90)
    iris = blue | warm
    count, labels, stats, _ = cv2.connectedComponentsWithStats(iris.astype(np.uint8), connectivity=8)
    mask_area = max(1, int(mask.sum()))
    blobs = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 12 or area > 0.04 * mask_area:
            continue
        short = max(1, min(width, height))
        if max(width, height) / short > 3.2:
            continue
        if area / max(1, width * height) < 0.28:
            continue
        blobs.append(i)
    if not blobs:
        return ""
    blue_blobs = [i for i in blobs if float(np.median(hue[labels == i])) >= 95]
    ranked = blue_blobs or blobs
    ranked = sorted(ranked, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:2]
    hues = np.concatenate([hue[labels == i] for i in ranked])
    median = float(np.median(hues))
    if 90 <= median <= 135:
        return "blue"
    if 35 <= median < 90:
        return "green"
    if 16 <= median < 35:
        return "yellow"
    if 5 <= median < 16:
        return "copper"
    return ""


def square_pad(rgb: np.ndarray) -> np.ndarray:
    """Pad to a square so CLIP's center crop keeps the head instead of cutting it off."""
    height, width = rgb.shape[:2]
    side = max(height, width)
    if height == width:
        return rgb
    canvas = np.full((side, side, 3), 127, dtype=np.uint8)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = rgb
    return canvas


def crop_rgb(bgr: np.ndarray, xyxy, pad: float = 0.06):
    height, width = bgr.shape[:2]
    if xyxy is None:
        return square_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    x1 = max(0, int(x1 - pad * box_w))
    y1 = max(0, int(y1 - pad * box_h))
    x2 = min(width, int(x2 + pad * box_w))
    y2 = min(height, int(y2 + pad * box_h))
    if x2 <= x1 or y2 <= y1:
        return square_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return square_pad(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def classify_crops(model, preprocess, text, crops: list[np.ndarray]) -> list[dict]:
    from PIL import Image

    batch = torch.stack([preprocess(Image.fromarray(crop)) for crop in crops])
    scale = model.logit_scale.exp()
    with torch.inference_mode():
        image_features = model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    rows = []
    group_probs = {}
    for name, (labels, features) in text.items():
        logits = scale * image_features @ features.T
        group_probs[name] = logits.softmax(dim=-1).cpu().numpy()
    for i in range(len(crops)):
        breed_p = group_probs["breed"][i]
        framing_p = group_probs["framing"][i]
        view_p = group_probs["view"][i]
        eyes_p = group_probs["eyes"][i]
        eye_colour_p = group_probs["eyes_colour"][i]
        age_p = group_probs["age"][i]
        posture_p = group_probs["posture"][i]
        breed_i = int(breed_p.argmax())
        order = np.argsort(breed_p)[::-1]
        second = int(order[1])
        framing = FRAMING[int(framing_p.argmax())][0]
        view = VIEWS[int(view_p.argmax())][0]
        eyes = EYES[int(eyes_p.argmax())][0]
        eyes_colour = EYE_COLOURS[int(eye_colour_p.argmax())][0]
        age = AGE[int(age_p.argmax())][0]
        posture = POSTURE[int(posture_p.argmax())][0]
        rows.append(
            {
                "breed": BREEDS[breed_i][0],
                "breed_score": float(breed_p[breed_i]),
                "breed_margin": float(breed_p[breed_i] - breed_p[second]),
                "breed_second": BREEDS[second][0],
                "framing": framing,
                "framing_score": float(framing_p.max()),
                "view": view,
                "view_score": float(view_p.max()),
                "camera_angle": f"{framing}_{view}",
                "eyes": eyes,
                "eyes_score": float(eyes_p.max()),
                "eyes_colour": "" if eyes in {"closed", "not_visible"} else eyes_colour,
                "eyes_colour_score": 0.0 if eyes in {"closed", "not_visible"} else float(eye_colour_p.max()),
                "eyes_colour_clip": eyes_colour,
                "eyes_colour_clip_score": float(eye_colour_p.max()),
                "age": age,
                "age_score": float(age_p.max()),
                "posture": posture,
                "posture_score": float(posture_p.max()),
            }
        )
    return rows


def weighted_mode(labels, weights) -> tuple[str, float]:
    totals: dict[str, float] = {}
    for label, weight in zip(labels, weights):
        if label is None or label == "" or (isinstance(label, float) and np.isnan(label)):
            continue
        totals[str(label)] = totals.get(str(label), 0.0) + float(weight)
    if not totals:
        return "", 0.0
    best = max(totals, key=lambda key: (totals[key], key))
    return best, totals[best] / sum(totals.values())


def split_angle(angle: str) -> tuple[str, str]:
    for framing, _ in sorted(FRAMING, key=lambda item: len(item[0]), reverse=True):
        prefix = framing + "_"
        if angle.startswith(prefix):
            return framing, angle[len(prefix) :]
    return "", ""


def reconcile_breed(row: dict) -> str:
    """Keep a specific breed only when CLIP is confident and the coat does not contradict it."""
    breed = row["breed"]
    domestic = {"domestic_shorthair", "domestic_longhair"}
    if breed in domestic:
        return breed
    light = row["share_white"] + row["share_cream"]
    second = row["breed_second"] if row["breed_second"] in domestic else "domestic_shorthair"
    if breed == "siamese" and row["colour_pattern"] != "pointed" and light < 0.40:
        return second
    if row["breed_margin"] < 0.42:
        return second
    return breed


def apply_framing(row: dict) -> None:
    """Override CLIP framing only when the box size makes the shot obvious."""
    area = row["bbox_area_frac"]
    edges = row["edges_touched"]
    if row["cat_detected"] and (area >= 0.72 or (edges >= 3 and area >= 0.40)):
        row["framing"] = "close_up"
    elif row["cat_detected"] and area <= 0.22 and edges <= 1:
        row["framing"] = "full_body"
    row["camera_angle"] = f"{row['framing']}_{row['view']}"


def eyes_description(eyes: str, eyes_colour: str) -> str:
    if eyes_colour and eyes == "both_open":
        return f"{eyes_colour}, both open"
    if eyes_colour and eyes == "half_closed":
        return f"{eyes_colour}, half closed"
    if eyes_colour:
        return eyes_colour
    if eyes == "closed":
        return "closed"
    if eyes == "not_visible":
        return "not visible"
    return eyes


def aggregate(images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, group in images.groupby("cat", sort=True):
        detected = group[group["cat_detected"] == 1]
        use = detected if len(detected) else group
        shares = {name: float(use[f"share_{name}"].mean()) for name in COLOURS}
        share_total = sum(shares.values()) or 1.0
        shares = {name: value / share_total for name, value in shares.items()}
        tabby = float(use["tabby_score"].mean())
        pointed = float(use["pointed_score"].mean())
        primary, secondary, pattern = labels_from_coat(shares, tabby, pointed)
        breed, breed_agreement = weighted_mode(use["breed"], use["breed_score"].clip(lower=0.01))
        voted = use[use["breed"] == breed]
        angle, camera_agreement = weighted_mode(
            use["camera_angle"],
            (use["framing_score"] * use["view_score"]).clip(lower=0.01),
        )
        framing, view = split_angle(angle)
        eyes, eyes_agreement = weighted_mode(use["eyes"], use["eyes_score"].clip(lower=0.01))
        age, age_agreement = weighted_mode(use["age"], use["age_score"].clip(lower=0.01))
        posture, posture_agreement = weighted_mode(use["posture"], use["posture_score"].clip(lower=0.01))
        open_rows = use[(~use["eyes"].isin(["closed", "not_visible"])) & (use["eyes_colour"] != "")]
        if len(open_rows):
            eyes_colour, eyes_colour_agreement = weighted_mode(
                open_rows["eyes_colour"], open_rows["eyes_colour_score"].clip(lower=0.01)
            )
        else:
            eyes_colour, eyes_colour_agreement = "", 0.0
        rows.append(
            {
                "cat": cat,
                "n_images": int(len(group)),
                "n_cat_detected": int(group["cat_detected"].sum()),
                "colour_primary": primary,
                "colour_secondary": secondary,
                "colour_pattern": pattern,
                **{f"share_{name}": shares[name] for name in COLOURS},
                "breed": breed,
                "breed_agreement": breed_agreement,
                "breed_score": float(voted["breed_score"].mean()) if len(voted) else 0.0,
                "breed_margin": float(voted["breed_margin"].mean()) if len(voted) else 0.0,
                "camera_angle": angle,
                "camera_agreement": camera_agreement,
                "framing": framing,
                "view": view,
                "eyes": eyes,
                "eyes_agreement": eyes_agreement,
                "eyes_open_share": float((group["eyes"] == "both_open").mean()),
                "eyes_colour": eyes_colour,
                "eyes_colour_agreement": eyes_colour_agreement,
                "eyes_description": eyes_description(eyes, eyes_colour),
                "age": age,
                "age_agreement": age_agreement,
                "posture": posture,
                "posture_agreement": posture_agreement,
            }
        )
    attributes = pd.DataFrame(rows)
    strength_path = PROCESSED / "cat_strengths.csv"
    if strength_path.exists():
        strengths = pd.read_csv(strength_path)
        attributes = attributes.merge(strengths, on="cat", how="left")
        for column in ("appearances", "wins"):
            attributes[column] = attributes[column].astype("Int64")
    return attributes.sort_values("cat").reset_index(drop=True)


def output_paths(tag: str) -> tuple[Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return PROCESSED / f"image_attributes{suffix}.csv", PROCESSED / f"cat_attributes{suffix}.csv"


def run(limit: int, sample: int, batch: int, imgsz: int, tag: str, fresh: bool) -> None:
    torch.set_num_threads(8)
    image_csv, cat_csv = output_paths(tag)
    paths = select_paths(list_images(), limit, sample)
    done: set[str] = set()
    if image_csv.exists() and not fresh:
        header = pd.read_csv(image_csv, nrows=0).columns
        if "age" not in header or "posture" not in header:
            print("Existing image CSV has no age or posture columns; starting fresh.", flush=True)
            fresh = True
    if image_csv.exists() and not fresh:
        previous = pd.read_csv(image_csv)
        done = set(previous["image"].astype(str))
        paths = [path for path in paths if f"{path.parent.name}/{path.name}" not in done]
        print(f"Resuming: {len(done)} photos already saved, {len(paths)} left", flush=True)
    elif image_csv.exists() and fresh:
        image_csv.unlink()

    if paths:
        print(f"Loading models for {len(paths)} photos", flush=True)
        yolo, _cat_ids = load_yolo()
        clip, preprocess, tokenizer = load_clip()
        text = prepare_text(clip, tokenizer)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        wrote_header = not image_csv.exists()
        for start in range(0, len(paths), batch):
            chunk = paths[start : start + batch]
            results = yolo.predict(
                [str(path) for path in chunk],
                imgsz=imgsz,
                conf=0.05,
                verbose=False,
                device="cpu",
                stream=False,
            )
            records = []
            crops = []
            regions = []
            for path, result in zip(chunk, results):
                bgr = result.orig_img
                inst = animal_instance(result)
                if inst is None:
                    mask = np.ones(bgr.shape[:2], dtype=bool)
                    record = {
                        "cat": path.parent.name,
                        "image": f"{path.parent.name}/{path.name}",
                        "cat_detected": 0,
                        "det_name": "",
                        "det_conf": 0.0,
                        "bbox_area_frac": 0.0,
                        "edges_touched": 0,
                        "colour_source": "full_frame",
                    }
                    crops.append(crop_rgb(bgr, None))
                else:
                    mask = inst["mask"]
                    record = {
                        "cat": path.parent.name,
                        "image": f"{path.parent.name}/{path.name}",
                        "cat_detected": 1,
                        "det_name": inst["name"],
                        "det_conf": inst["conf"],
                        "bbox_area_frac": inst["area_frac"],
                        "edges_touched": edges_touched(mask),
                        "colour_source": inst["source"],
                    }
                    crops.append(crop_rgb(bgr, inst["xyxy"]))
                record.update(coat_features(bgr, mask))
                records.append(record)
                regions.append((bgr, mask))
            for record, labels, (bgr, mask) in zip(records, classify_crops(clip, preprocess, text, crops), regions):
                record.update(labels)
                record["breed_raw"] = record["breed"]
                record["breed"] = reconcile_breed(record)
                apply_framing(record)
                if record["eyes"] in {"closed", "not_visible"}:
                    blob = eye_colour_from_mask(bgr, mask)
                    if blob:
                        record["eyes"] = "both_open"
                        if record["eyes_colour_clip_score"] >= 0.60:
                            record["eyes_colour"] = record["eyes_colour_clip"]
                            record["eyes_colour_score"] = record["eyes_colour_clip_score"]
                        else:
                            record["eyes_colour"] = blob
                            record["eyes_colour_score"] = 0.5
                record.pop("eyes_colour_clip", None)
                record.pop("eyes_colour_clip_score", None)
            frame = pd.DataFrame(records)
            frame.to_csv(image_csv, mode="a", header=wrote_header, index=False)
            wrote_header = False
            print(f"{min(start + batch, len(paths))}/{len(paths)}", flush=True)

    if not image_csv.exists():
        raise SystemExit("No image rows were written.")
    images = pd.read_csv(image_csv).fillna("")
    cats = aggregate(images)
    skip = {"appearances", "wins", "strength", "n_images", "n_cat_detected"}
    float_cols = [col for col in cats.select_dtypes(include="number").columns if col not in skip]
    cats[float_cols] = cats[float_cols].round(4)
    cats.to_csv(cat_csv, index=False, na_rep="")
    print(f"Wrote {image_csv} ({len(images)} photos) and {cat_csv} ({len(cats)} cats)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label breed, colour, camera angle, eyes, age, and posture for each cat.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N photos, in filename order.")
    parser.add_argument("--sample", type=int, default=0, help="One photo from this many evenly spaced cats.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--tag", default="", help="Write image_attributes_<tag>.csv instead of the main files.")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing image CSV and start again.")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild the per-cat CSV from the image CSV.")
    args = parser.parse_args()
    if args.aggregate_only:
        image_csv, cat_csv = output_paths(args.tag)
        images = pd.read_csv(image_csv).fillna("")
        cats = aggregate(images)
        skip = {"appearances", "wins", "strength", "n_images", "n_cat_detected"}
        float_cols = [col for col in cats.select_dtypes(include="number").columns if col not in skip]
        cats[float_cols] = cats[float_cols].round(4)
        cats.to_csv(cat_csv, index=False, na_rep="")
        print(f"Wrote {cat_csv} ({len(cats)} cats)", flush=True)
        return
    run(args.limit, args.sample, args.batch, args.imgsz, args.tag, args.fresh)


if __name__ == "__main__":
    main()
