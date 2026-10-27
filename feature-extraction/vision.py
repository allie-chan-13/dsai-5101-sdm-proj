"""YOLO masks, coat colour, and CLIP labels."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch

from main import (
    AGE,
    BREED_MARGIN,
    BREEDS,
    CAT_CLASS_NAME,
    CAT_CONF,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    CLIP_QUICK_GELU,
    CLOSE_UP_AREA,
    CLOSE_UP_EDGE_AREA,
    CLOSE_UP_EDGES,
    COLOURS,
    CROP_PAD,
    DOG_CONF,
    EYE_COLOURS,
    EYES,
    FRAMING,
    FULL_BODY_AREA,
    FULL_BODY_EDGES,
    IMAGE_ROOT,
    POSTURE,
    PROMPT_GROUPS,
    TEMPLATES,
    VIEWS,
    WEIGHTS,
)

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

    logging.getLogger("open_clip").setLevel(logging.ERROR)

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
        force_quick_gelu=CLIP_QUICK_GELU,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
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
    cats = [item for item in candidates if item[0] == "cat" and item[1] >= CAT_CONF]
    dogs = [item for item in candidates if item[0] == "dog" and item[1] >= DOG_CONF]
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


def box_pixels(xyxy, height: int, width: int) -> dict[str, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return {
        "bbox_x1": int(np.clip(round(x1), 0, max(0, width - 1))),
        "bbox_y1": int(np.clip(round(y1), 0, max(0, height - 1))),
        "bbox_x2": int(np.clip(round(x2), 0, max(0, width - 1))),
        "bbox_y2": int(np.clip(round(y2), 0, max(0, height - 1))),
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


def open_eye_score(bgr: np.ndarray) -> tuple[float, float]:
    """How strongly a crop shows a catchlight inside a dark pupil.

    The first value is the strongest single eye. The second is the weaker eye
    of a horizontal pair. Both are 0 when no pupil is found. Shut eyes and
    pale irises without a dark centre stay at 0.
    """
    height, width = bgr.shape[:2]
    if height < 20 or width < 20:
        return 0.0, 0.0
    scale = 900 / max(height, width)
    image = bgr if scale >= 1 else cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape
    windows = [(0, 0, frame_w, frame_h)]
    for y_frac in (0.0, 0.35):
        for x_frac in (0.0, 0.35):
            windows.append(
                (
                    int(frame_w * x_frac),
                    int(frame_h * y_frac),
                    int(frame_w * min(1, x_frac + 0.7)),
                    int(frame_h * min(1, y_frac + 0.7)),
                )
            )
    best_single, best_pair = 0.0, 0.0
    for x0, y0, x1, y1 in windows:
        pupils = _pupil_catchlights(gray[y0:y1, x0:x1])
        if pupils:
            best_single = max(best_single, pupils[0][0])
        patch_h, patch_w = y1 - y0, x1 - x0
        for i, left in enumerate(pupils):
            for right in pupils[i + 1 :]:
                dx = abs(left[1] - right[1])
                dy = abs(left[2] - right[2])
                if dy > max(14, 0.1 * patch_h) or dx < 0.08 * patch_w or dx > 0.62 * patch_w or dy > 0.5 * dx:
                    continue
                best_pair = max(best_pair, min(left[0], right[0]))
    return best_single, best_pair


def _pupil_catchlights(gray: np.ndarray) -> list[tuple[float, int, int]]:
    height, width = gray.shape
    if height < 30 or width < 30:
        return []
    blur = cv2.GaussianBlur(gray, (0, 0), 8)
    bright = ((gray.astype(np.int16) - blur.astype(np.int16) > 40) & (gray > 160)).astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    found = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        box_w = int(stats[i, cv2.CC_STAT_WIDTH])
        box_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 1 or area > 30 or max(box_w, box_h) > 8:
            continue
        x = int(round(centroids[i][0]))
        y = int(round(centroids[i][1]))
        if not (4 < x < width - 4 and 4 < y < height - 4):
            continue
        best = None
        for radius in (8, 12, 18, 26, 36):
            reach = int(radius * 1.8)
            y0, y1 = max(0, y - reach), min(height, y + reach + 1)
            x0, x1 = max(0, x - reach), min(width, x + reach + 1)
            patch = gray[y0:y1, x0:x1]
            yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
            dist = np.sqrt((yy - (y - y0)) ** 2 + (xx - (x - x0)) ** 2)
            core = (dist >= radius * 0.35) & (dist <= radius * 0.85)
            outer = (dist >= radius * 1.05) & (dist <= radius * 1.7)
            if int(core.sum()) < 8 or int(outer.sum()) < 8:
                continue
            core_mean = float(patch[core].mean())
            outer_mean = float(patch[outer].mean())
            spot = float(gray[y, x])
            if core_mean > 45 or spot - core_mean < 80 or outer_mean - core_mean < 15:
                continue
            score = (spot - core_mean) + (outer_mean - core_mean) - core_mean
            if best is None or score > best[0]:
                best = (score, x, y)
        if best:
            found.append(best)
    found.sort(reverse=True)
    kept = []
    for item in found:
        if any((item[1] - other[1]) ** 2 + (item[2] - other[2]) ** 2 < 625 for other in kept):
            continue
        kept.append(item)
        if len(kept) == 6:
            break
    return kept


def _field(row, name):
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name)


def crop_box(bgr: np.ndarray, row) -> np.ndarray:
    height, width = bgr.shape[:2]
    values = [_field(row, name) for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
    if any(value is None or value is pd.NA or str(value).strip() in {"", "nan", "<NA>", "None"} for value in values):
        return bgr
    x1, y1, x2, y2 = [int(float(value)) for value in values]
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return bgr
    return bgr[y1:y2, x1:x2]


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


def crop_rgb(bgr: np.ndarray, xyxy, pad: float = CROP_PAD):
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
    if row['breed_margin'] < BREED_MARGIN:
        return second
    return breed


def apply_framing(row: dict) -> None:
    """Override CLIP framing only when the box size makes the shot obvious."""
    area = row["bbox_area_frac"]
    edges = row["edges_touched"]
    if row["cat_detected"] and (area >= CLOSE_UP_AREA or (edges >= CLOSE_UP_EDGES and area >= CLOSE_UP_EDGE_AREA)):
        row["framing"] = "close_up"
    elif row["cat_detected"] and area <= FULL_BODY_AREA and edges <= FULL_BODY_EDGES:
        row["framing"] = "full_body"
    row["camera_angle"] = f"{row['framing']}_{row['view']}"
