"""Write the attribute tables and the captioned photos."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch

from main import (
    AWAY_VIEWS,
    BOX_FIELDS,
    CLOSED_TO_OPEN_SCORE,
    COLOURS,
    DROP_COLUMNS,
    EYE_COLOUR_BLOB,
    EYE_COLOUR_MIN,
    FACE_FRAMING,
    FACE_VIEWS,
    IMAGE_ROOT,
    LABEL_FONT,
    LABEL_JPEG_QUALITY,
    OPEN_EYE_PAIR,
    OPEN_EYE_SCORE,
    OPEN_EYE_SINGLE,
    PROCESSED,
    ROOT,
    THREADS,
    YOLO_CONF,
    YOLO_DEVICE,
)
from vision import (
    animal_instance,
    apply_framing,
    box_pixels,
    classify_crops,
    coat_features,
    crop_box,
    crop_rgb,
    edges_touched,
    eye_colour_from_mask,
    labels_from_coat,
    list_images,
    load_clip,
    load_yolo,
    open_eye_score,
    prepare_text,
    reconcile_breed,
    select_paths,
    split_angle,
    weighted_mode,
)

EMPTY_BOX = {name: pd.NA for name in BOX_FIELDS}
BOX_COLUMNS = set(BOX_FIELDS)
IMAGE_INT = {"cat_detected", "edges_touched", *BOX_COLUMNS}
CAT_INT = {"n_images", "n_cat_detected", "appearances", "wins"}

def apply_open_eyes(images: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set eyes to both_open where a dark pupil with a catchlight is visible."""
    changed = 0
    eyes = images["eyes"].astype(str).tolist()
    scores = images["eyes_score"].tolist()
    for i, row in enumerate(images.itertuples(index=False)):
        image = cv2.imread(str(IMAGE_ROOT / row.image))
        if image is None:
            continue
        single, pair = open_eye_score(crop_box(image, row))
        face = row.view in FACE_VIEWS and row.framing in FACE_FRAMING
        if single >= OPEN_EYE_SINGLE or pair >= OPEN_EYE_PAIR or (eyes[i] == "half_closed" and face):
            if eyes[i] != "both_open":
                changed += 1
            eyes[i] = "both_open"
            scores[i] = round(max(float(scores[i]) if str(scores[i]) not in {"", "nan"} else 0.0, OPEN_EYE_SCORE), 4)
    images = images.copy()
    images["eyes"] = eyes
    images["eyes_score"] = scores
    return images, changed


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
        if "age" in use.columns and "posture" in use.columns:
            age, age_agreement = weighted_mode(use["age"], use["age_score"].clip(lower=0.01))
            posture, posture_agreement = weighted_mode(use["posture"], use["posture_score"].clip(lower=0.01))
            age_fields = {
                "age": age,
                "age_agreement": age_agreement,
                "posture": posture,
                "posture_agreement": posture_agreement,
            }
        else:
            age_fields = {}
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
                **age_fields,
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


def tidy_frame(frame: pd.DataFrame, keep_int: set[str]) -> pd.DataFrame:
    frame = frame.drop(columns=[name for name in DROP_COLUMNS if name in frame.columns])
    numeric = [name for name in frame.select_dtypes(include="number").columns if name not in keep_int]
    frame[numeric] = frame[numeric].round(4)
    for name in keep_int:
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name]):
            frame[name] = frame[name].astype("Int64")
    text = [name for name in frame.columns if name not in numeric and name not in keep_int]
    if text:
        frame[text] = frame[text].fillna("")
    return frame


def _text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "none"}:
        return ""
    return text.replace("_", " ")


def caption_lines(row: dict) -> list[str]:
    primary = _text(row.get("colour_primary"))
    secondary = _text(row.get("colour_secondary"))
    pattern = _text(row.get("colour_pattern"))
    colour = primary if not secondary else f"{primary} / {secondary}"
    if pattern:
        colour = f"{colour} {pattern}".strip()
    eyes = _text(row.get("eyes"))
    eyes_colour = _text(row.get("eyes_colour"))
    eye_line = eyes if not eyes_colour else f"{eyes_colour}, {eyes}"
    pose = ", ".join(part for part in (_text(row.get("framing")), _text(row.get("view"))) if part)
    lines = [colour, _text(row.get("breed")), pose, eye_line]
    for key in ("age", "posture"):
        extra = _text(row.get(key))
        if extra:
            lines.append(extra)
    return [line for line in lines if line]


def _box(row: dict) -> tuple[int, int, int, int] | None:
    values = []
    for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        raw = row.get(key, "")
        if raw is None or (isinstance(raw, float) and raw != raw):
            return None
        text = str(raw).strip()
        if not text or text.lower() in {"nan", "<na>", "none"}:
            return None
        try:
            values.append(int(float(text)))
        except ValueError:
            return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def export_labelled_images(images: pd.DataFrame) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    font_path = LABEL_FONT
    out_root = PROCESSED / "labelled"
    written = 0
    for row in images.to_dict("records"):
        src = IMAGE_ROOT / row["image"]
        if not src.is_file():
            continue
        photo = Image.open(src).convert("RGB")
        font = ImageFont.truetype(str(font_path), max(18, photo.width // 42)) if font_path.is_file() else ImageFont.load_default()
        box = _box(row)
        if box:
            stroke = max(3, photo.width // 180)
            boxed = ImageDraw.Draw(photo)
            boxed.rectangle(box, outline=(0, 0, 0), width=stroke + 2)
            boxed.rectangle(box, outline=(255, 214, 10), width=stroke)
        lines = caption_lines(row)
        scratch = ImageDraw.Draw(photo)
        line_h = max(scratch.textbbox((0, 0), line, font=font)[3] for line in lines) + 4
        pad = 10
        bar_h = pad * 2 + line_h * len(lines)
        canvas = Image.new("RGB", (photo.width, photo.height + bar_h), (24, 24, 24))
        canvas.paste(photo, (0, 0))
        draw = ImageDraw.Draw(canvas)
        y = photo.height + pad
        for line in lines:
            draw.text((pad, y), line, fill=(245, 245, 245), font=font)
            y += line_h
        dest = out_root / row["image"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, quality=LABEL_JPEG_QUALITY)
        written += 1
    print(f"Wrote {written} labelled images to {out_root.relative_to(ROOT)}", flush=True)
    return out_root


def output_paths(tag: str) -> tuple[Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return PROCESSED / f"image_attributes{suffix}.csv", PROCESSED / f"cat_attributes{suffix}.csv"


def run(limit: int, sample: int, batch: int, imgsz: int, tag: str, fresh: bool) -> None:
    torch.set_num_threads(THREADS)
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
                conf=YOLO_CONF,
                verbose=False,
                device=YOLO_DEVICE,
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
                        **EMPTY_BOX,
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
                        **box_pixels(inst["xyxy"], bgr.shape[0], bgr.shape[1]),
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
                single, pair = open_eye_score(crop_box(bgr, record))
                face = record["view"] in FACE_VIEWS and record["framing"] in FACE_FRAMING
                if single >= OPEN_EYE_SINGLE or pair >= OPEN_EYE_PAIR or (record["eyes"] == "half_closed" and face):
                    record["eyes"] = "both_open"
                    record["eyes_score"] = max(float(record['eyes_score']), OPEN_EYE_SCORE)
                    if not record["eyes_colour"] and record['eyes_colour_clip_score'] >= EYE_COLOUR_MIN:
                        record["eyes_colour"] = record["eyes_colour_clip"]
                        record["eyes_colour_score"] = record["eyes_colour_clip_score"]
                elif record["eyes"] == "closed" and record['view'] in AWAY_VIEWS:
                    record["eyes"] = "not_visible"
                    record["eyes_colour"] = ""
                    record["eyes_colour_score"] = 0.0
                elif record["eyes"] == "closed":
                    record["eyes"] = "both_open"
                    record["eyes_score"] = max(float(record['eyes_score']), CLOSED_TO_OPEN_SCORE)
                if record["eyes"] in {"closed", "not_visible"}:
                    blob = eye_colour_from_mask(bgr, mask)
                    if blob:
                        record["eyes"] = "both_open"
                        if record['eyes_colour_clip_score'] >= EYE_COLOUR_BLOB:
                            record["eyes_colour"] = record["eyes_colour_clip"]
                            record["eyes_colour_score"] = record["eyes_colour_clip_score"]
                        else:
                            record["eyes_colour"] = blob
                            record["eyes_colour_score"] = 0.5
                record.pop("eyes_colour_clip", None)
                record.pop("eyes_colour_clip_score", None)
            frame = tidy_frame(pd.DataFrame(records), IMAGE_INT)
            frame.to_csv(image_csv, mode="a", header=wrote_header, index=False, na_rep="")
            wrote_header = False
            done_n = min(start + batch, len(paths))
            if done_n == len(paths) or done_n % 200 < batch:
                print(f"{done_n}/{len(paths)}", flush=True)

    if not image_csv.exists():
        raise SystemExit("No image rows were written.")
    images = pd.read_csv(image_csv).fillna("")
    cats = aggregate(images)
    cats = tidy_frame(cats, CAT_INT)
    cats.to_csv(cat_csv, index=False, na_rep="")
    export_labelled_images(images)
    print(f"Wrote {image_csv} ({len(images)} photos) and {cat_csv} ({len(cats)} cats)", flush=True)


def fill_boxes(batch: int, imgsz: int, tag: str) -> None:
    """Run YOLO only, store the box, and redraw the labelled photos."""
    torch.set_num_threads(THREADS)
    image_csv, _ = output_paths(tag)
    images = pd.read_csv(image_csv)
    present = []
    for rel in images["image"].astype(str):
        path = IMAGE_ROOT / rel
        if path.is_file():
            present.append(path)
    print(f"Loading YOLO for boxes on {len(present)} photos", flush=True)
    yolo, _cat_ids = load_yolo()
    found: dict[str, dict[str, int]] = {}
    for start in range(0, len(present), batch):
        chunk = present[start : start + batch]
        results = yolo.predict(
            [str(path) for path in chunk],
            imgsz=imgsz,
            conf=YOLO_CONF,
            verbose=False,
            device=YOLO_DEVICE,
            stream=False,
        )
        for path, result in zip(chunk, results):
            inst = animal_instance(result)
            if inst is None:
                continue
            height, width = result.orig_shape
            found[f"{path.parent.name}/{path.name}"] = box_pixels(inst["xyxy"], height, width)
        done_n = min(start + batch, len(present))
        if done_n == len(present) or done_n % 200 < batch:
            print(f"{done_n}/{len(present)}", flush=True)
    ordered = [name for name in images.columns if name not in BOX_COLUMNS]
    insert_at = ordered.index("bbox_area_frac") + 1 if "bbox_area_frac" in ordered else len(ordered)
    for offset, name in enumerate(EMPTY_BOX):
        ordered.insert(insert_at + offset, name)
        images[name] = [found.get(rel, {}).get(name, pd.NA) for rel in images["image"].astype(str)]
    images = images[ordered]
    images = tidy_frame(images, IMAGE_INT)
    images.to_csv(image_csv, index=False, na_rep="")
    print(f"Boxes on {len(found)} of {len(images)} photos", flush=True)
    export_labelled_images(images)
