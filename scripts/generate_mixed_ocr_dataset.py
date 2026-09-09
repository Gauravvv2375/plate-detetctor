"""Generate a deterministic Latin + Devanagari synthetic ANPR OCR dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASCII_DIGITS = "0123456789"
DEVANAGARI_DIGITS = "०१२३४५६७८९"
DIGIT_TRANSLATION = str.maketrans(ASCII_DIGITS, DEVANAGARI_DIGITS)
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
STATE_CODES = (
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA", "GJ", "HP", "HR", "JH",
    "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
)
LETTER_NAMES = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी", "H": "एच",
    "I": "आई", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी",
    "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू",
    "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}
HEADER_TEXTS = ("महाराष्ट्र", "भारत", "परिवहन", "महाराष्ट्राची शान", "INDIA", "भारत सरकार")
MIXED_SAFE_FONT_SPECS = (
    # Verified by cmap audit to contain the complete Latin + Devanagari
    # vocabulary used by this generator. Pillow does not perform font
    # fallback, so a Devanagari-only face renders Latin letters as identical
    # .notdef boxes and must never be used for mixed OCR ground truth.
    (Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), 0, "Arial Unicode"),
    (Path("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"), 0, "Devanagari Sangam MN 0"),
    (Path("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"), 1, "Devanagari Sangam MN 1"),
)
LEGACY_UNSAFE_MIXED_FONT_SPECS = (
    (Path("/System/Library/Fonts/Supplemental/DevanagariMT.ttc"), 0, "Devanagari MT 0"),
    (Path("/System/Library/Fonts/Supplemental/DevanagariMT.ttc"), 1, "Devanagari MT 1"),
    (Path("/System/Library/Fonts/Supplemental/ITFDevanagari.ttc"), 0, "ITF Devanagari 0"),
    (Path("/System/Library/Fonts/Supplemental/ITFDevanagari.ttc"), 2, "ITF Devanagari 2"),
    (Path("/System/Library/Fonts/Supplemental/ITFDevanagari.ttc"), 5, "ITF Devanagari 5"),
)
FONT_AUDIT_SPECS = MIXED_SAFE_FONT_SPECS + LEGACY_UNSAFE_MIXED_FONT_SPECS
FONT_SPECS = MIXED_SAFE_FONT_SPECS
CATEGORIES = (
    "latin_only",
    "devanagari_only",
    "latin_letters_devanagari_digits",
    "devanagari_letters_ascii_digits",
    "mixed_script",
)
FIXED_EXAMPLES = (
    ("MH12AB1234", "latin_only", "MH", "12", "AB", "1234"),
    ("एमएच१२एबी१२३४", "devanagari_only", "MH", "12", "AB", "1234"),
    ("MH १२ AB १२३४", "latin_letters_devanagari_digits", "MH", "12", "AB", "1234"),
    ("एमएच 12 एबी 1234", "devanagari_letters_ascii_digits", "MH", "12", "AB", "1234"),
    ("MH 12 एबी १२३४", "mixed_script", "MH", "12", "AB", "1234"),
    ("एमएच १२ AB 1234", "mixed_script", "MH", "12", "AB", "1234"),
    ("MH-१२-AB-१२३४", "latin_letters_devanagari_digits", "MH", "12", "AB", "1234"),
    ("DL०१CA१२३४", "latin_letters_devanagari_digits", "DL", "01", "CA", "1234"),
    ("केए05MN६७८९", "mixed_script", "KA", "05", "MN", "6789"),
)
CSV_FIELDS = (
    "image", "text", "category", "layout", "difficulty", "split", "group_id", "state", "district",
    "series", "registration", "separator", "font", "background", "header_text", "render_sha256",
)


def devanagari_letters(text: str) -> str:
    return "".join(LETTER_NAMES[character] for character in text)


def devanagari_digits(text: str) -> str:
    return text.translate(DIGIT_TRANSLATION)


def choose_scripts(category: str, rng: random.Random) -> tuple[bool, bool, bool, bool]:
    """Return Devanagari flags for state, district, series, registration groups."""
    if category == "latin_only":
        return False, False, False, False
    if category == "devanagari_only":
        return True, True, True, True
    if category == "latin_letters_devanagari_digits":
        return False, True, False, True
    if category == "devanagari_letters_ascii_digits":
        return True, False, True, False
    choices = rng.choice(
        (
            (False, True, True, False),
            (True, False, False, True),
            (False, True, True, True),
            (True, True, False, False),
        )
    )
    return choices


def registration_parts(index: int, rng: random.Random, category: str) -> tuple[list[str], dict[str, str]]:
    state = STATE_CODES[index % len(STATE_CODES)]
    district = f"{1 + (index * 17 + rng.randrange(99)) % 99:02d}"
    series_length = 1 if rng.random() < 0.18 else 2
    series = "".join(chr(65 + rng.randrange(26)) for _ in range(series_length))
    registration = f"{(index * 7919 + rng.randrange(10000)) % 10000:04d}"
    state_dev, district_dev, series_dev, registration_dev = choose_scripts(category, rng)
    parts = [
        devanagari_letters(state) if state_dev else state,
        devanagari_digits(district) if district_dev else district,
        devanagari_letters(series) if series_dev else series,
        devanagari_digits(registration) if registration_dev else registration,
    ]
    raw = {"state": state, "district": district, "series": series, "registration": registration}
    return parts, raw


def deterministic_split(group_id: str, val_fraction: float) -> str:
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    return "val" if value < val_fraction else "train"


def available_fonts() -> list[tuple[Path, int, str]]:
    fonts = []
    for path, index, name in FONT_SPECS:
        if not path.is_file():
            continue
        try:
            ImageFont.truetype(str(path), 32, index=index)
        except OSError:
            continue
        fonts.append((path, index, name))
    if not fonts:
        raise RuntimeError("No mixed Latin/Devanagari font was found on this system")
    return fonts


def fitted_font(
    text: str,
    font_spec: tuple[Path, int, str],
    maximum_width: int,
    maximum_height: int,
    preferred_size: int,
) -> ImageFont.FreeTypeFont:
    path, index, _ = font_spec
    for size in range(preferred_size, 11, -1):
        font = ImageFont.truetype(str(path), size, index=index)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= maximum_width and bottom - top <= maximum_height:
            return font
    return ImageFont.truetype(str(path), 12, index=index)


def centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    stroke_width: int,
    shadow: bool,
) -> None:
    left, top, right, bottom = bounds
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    x = left + (right - left - (box[2] - box[0])) / 2 - box[0]
    y = top + (bottom - top - (box[3] - box[1])) / 2 - box[1]
    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=(80, 80, 80), stroke_width=stroke_width)
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width)


def render_plate(
    label: str,
    visual_rows: list[str],
    layout: str,
    header_text: str,
    difficulty: str,
    font_spec: tuple[Path, int, str],
    rng: random.Random,
) -> tuple[Image.Image, str]:
    if layout == "one_line":
        width, height = rng.randint(310, 540), rng.randint(74, 116)
    else:
        width, height = rng.randint(250, 390), rng.randint(132, 204)
    yellow = rng.random() < 0.20
    background_name = "yellow" if yellow else "white"
    base = (238, 194, 52) if yellow else (rng.randint(224, 250),) * 3
    image = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(image)
    border = rng.randint(2, 5)
    draw.rounded_rectangle((border, border, width - border - 1, height - border - 1), radius=rng.randint(2, 9), outline=(25, 25, 25), width=border)
    ink_level = rng.randint(8, 38) if difficulty != "difficult" else rng.randint(28, 72)
    ink = (ink_level, ink_level, ink_level)
    stroke_width = rng.choice((0, 0, 1, 1, 2))
    shadow = difficulty == "difficult" and rng.random() < 0.25
    margin_x, margin_y = max(9, width // 30), max(8, height // 24)

    if layout == "one_line":
        font = fitted_font(label, font_spec, width - 2 * margin_x, height - 2 * margin_y, rng.randint(42, 66))
        centered_text(draw, (margin_x, margin_y, width - margin_x, height - margin_y), label, font, ink, stroke_width, shadow)
    elif layout == "two_line":
        row_height = (height - 2 * margin_y) // 2
        for row_index, row_text in enumerate(visual_rows):
            top = margin_y + row_index * row_height
            font = fitted_font(row_text, font_spec, width - 2 * margin_x, row_height - 3, rng.randint(36, 54))
            centered_text(draw, (margin_x, top, width - margin_x, top + row_height), row_text, font, ink, stroke_width, shadow)
    else:
        header_height = max(24, round(height * 0.31))
        header_font = fitted_font(header_text, font_spec, width - 2 * margin_x, header_height - 6, rng.randint(17, 27))
        centered_text(draw, (margin_x, margin_y, width - margin_x, header_height), header_text, header_font, ink, 0, False)
        main_font = fitted_font(label, font_spec, width - 2 * margin_x, height - header_height - margin_y, rng.randint(42, 62))
        centered_text(draw, (margin_x, header_height, width - margin_x, height - margin_y), label, main_font, ink, stroke_width, shadow)

    if difficulty in {"medium", "difficult"}:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for _ in range(rng.randint(5, 22 if difficulty == "difficult" else 12)):
            x, y = rng.randrange(width), rng.randrange(height)
            radius = rng.randint(1, 4)
            overlay_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(70, 55, 35, rng.randint(5, 24)))
        if rng.random() < (0.40 if difficulty == "difficult" else 0.18):
            glare_x = rng.randint(width // 5, 4 * width // 5)
            overlay_draw.polygon(
                ((glare_x - width // 8, 0), (glare_x + width // 12, 0), (glare_x + width // 5, height), (glare_x, height)),
                fill=(255, 255, 255, rng.randint(15, 48)),
            )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.76, 1.13))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.10))
    if difficulty == "difficult":
        if rng.random() < 0.75:
            image = image.rotate(rng.uniform(-3.5, 3.5), Image.Resampling.BICUBIC, expand=False, fillcolor=base)
        if rng.random() < 0.65:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.35, 1.15)))
        if rng.random() < 0.50:
            small_width = max(96, round(width * rng.uniform(0.35, 0.68)))
            small_height = max(28, round(height * small_width / width))
            image = image.resize((small_width, small_height), Image.Resampling.BILINEAR).resize((width, height), Image.Resampling.BILINEAR)
        if rng.random() < 0.45:
            array = np.asarray(image, dtype=np.int16)
            noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2.0, 7.0), array.shape)
            image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        if rng.random() < 0.60:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=rng.randint(35, 72), subsampling=2)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
    elif difficulty == "medium":
        if rng.random() < 0.35:
            image = image.rotate(rng.uniform(-1.8, 1.8), Image.Resampling.BICUBIC, expand=False, fillcolor=base)
        if rng.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))
    return image, background_name


def sample_definition(index: int, seed: int, val_fraction: float, smoke: bool = False) -> dict[str, object]:
    rng = random.Random(seed + index * 1_000_003)
    if index <= len(FIXED_EXAMPLES):
        label, category, state, district, series, registration = FIXED_EXAMPLES[index - 1]
        layout = "two_line" if index in {2, 5} else "one_line"
        fixed_rows = {2: ["एमएच१२", "एबी१२३४"], 5: ["MH 12", "एबी १२३४"]}
        visual_rows = fixed_rows.get(index, [label])
        group_id = f"fixed-{state}-{district}-{series}-{registration}"
        return {
            "index": index,
            "label": unicodedata.normalize("NFC", label),
            "category": category,
            "layout": layout,
            "header_text": "",
            "visual_rows": visual_rows,
            "difficulty": ("clean", "medium", "difficult")[index % 3],
            "separator": "-" if "-" in label else " " if " " in label else "",
            "group_id": group_id,
            "split": deterministic_split(group_id, val_fraction),
            "state": state,
            "district": district,
            "series": series,
            "registration": registration,
        }
    if index == len(FIXED_EXAMPLES) + 1:
        label = "२३५६"
        group_id = "fixed-header-2356"
        return {
            "index": index,
            "label": label,
            "category": "devanagari_only",
            "layout": "header_registration",
            "header_text": "महाराष्ट्राची शान",
            "visual_rows": [label],
            "difficulty": "medium",
            "separator": "",
            "group_id": group_id,
            "split": deterministic_split(group_id, val_fraction),
            "state": "MH",
            "district": "",
            "series": "",
            "registration": "2356",
        }
    category = CATEGORIES[(index - 1) % len(CATEGORIES)]
    parts, raw = registration_parts(index, rng, category)
    separator = rng.choices(("", " ", "-"), weights=(0.32, 0.46, 0.22))[0]
    label = unicodedata.normalize("NFC", separator.join(parts))
    layout = rng.choices(("one_line", "two_line", "header_registration"), weights=(0.66, 0.24, 0.10))[0]
    header_text = rng.choice(HEADER_TEXTS) if layout == "header_registration" else ""
    visual_rows = [separator.join(parts[:2]), separator.join(parts[2:])] if layout == "two_line" else [label]
    difficulty = rng.choices(("clean", "medium", "difficult"), weights=(0.30, 0.45, 0.25))[0]
    group_id = f"{raw['state']}-{raw['district']}-{raw['series']}-{raw['registration']}"
    return {
        "index": index,
        "label": label,
        "category": category,
        "layout": layout,
        "header_text": header_text,
        "visual_rows": visual_rows,
        "difficulty": difficulty,
        "separator": separator,
        "group_id": group_id,
        "split": deterministic_split(group_id, val_fraction),
        **raw,
    }


def validate_label(label: str) -> list[str]:
    errors = []
    if not label:
        errors.append("empty label")
    if label != unicodedata.normalize("NFC", label):
        errors.append("label is not NFC")
    if any(character in ZERO_WIDTH for character in label):
        errors.append("zero-width character")
    if any(unicodedata.category(character).startswith("C") for character in label):
        errors.append("control/format character")
    if any(not (character in " -" or "0" <= character <= "9" or "A" <= character <= "Z" or "\u0900" <= character <= "\u097f") for character in label):
        errors.append("unsupported character")
    return errors


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new directory.")
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    fonts = available_fonts()
    rows: list[dict[str, str]] = []
    label_errors: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    seen_groups: dict[str, str] = {}
    group_splits: dict[str, set[str]] = {}
    label_splits: dict[str, set[str]] = {}
    count = args.count
    for index in range(1, count + 1):
        definition = None
        label = ""
        for collision_attempt in range(20):
            definition = sample_definition(
                index,
                args.seed + collision_attempt * 104_729,
                args.val_fraction,
                args.smoke,
            )
            label = str(definition["label"])
            if label not in seen_labels:
                break
        else:
            raise RuntimeError(f"Could not generate a unique label for sample {index}")
        assert definition is not None
        errors = validate_label(label)
        if errors:
            label_errors.append({"index": index, "label": label, "errors": errors})
            continue
        rng = random.Random(args.seed + index * 1_000_003 + 97)
        font_spec = fonts[(index - 1) % len(fonts)]
        image, background = render_plate(
            label,
            list(definition["visual_rows"]),
            str(definition["layout"]),
            str(definition["header_text"]),
            str(definition["difficulty"]),
            font_spec,
            rng,
        )
        relative_path = Path("images") / f"plate_{index:06d}.jpg"
        destination = output / relative_path
        image.save(destination, format="JPEG", quality=92, optimize=False)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        row = {
            "image": relative_path.as_posix(),
            "text": label,
            "category": str(definition["category"]),
            "layout": str(definition["layout"]),
            "difficulty": str(definition["difficulty"]),
            "split": str(definition["split"]),
            "group_id": str(definition["group_id"]),
            "state": str(definition["state"]),
            "district": str(definition["district"]),
            "series": str(definition["series"]),
            "registration": str(definition["registration"]),
            "separator": str(definition["separator"]),
            "font": font_spec[2],
            "background": background,
            "header_text": str(definition["header_text"]),
            "render_sha256": digest,
        }
        rows.append(row)
        seen_labels.add(label)
        prior_split = seen_groups.setdefault(row["group_id"], row["split"])
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
        label_splits.setdefault(row["text"], set()).add(row["split"])
        if prior_split != row["split"]:
            label_errors.append({"index": index, "label": label, "errors": ["group split leakage"]})
        if index % max(1, min(5000, count // 10)) == 0 or index == count:
            print(f"Rendered {index}/{count}")
    present_splits = {row["split"] for row in rows}
    if rows and "val" not in present_splits:
        forced_group = rows[-1]["group_id"]
        for row in rows:
            if row["group_id"] == forced_group:
                row["split"] = "val"
        seen_groups[forced_group] = "val"
    if rows and "train" not in present_splits:
        forced_group = rows[-1]["group_id"]
        for row in rows:
            if row["group_id"] == forced_group:
                row["split"] = "train"
        seen_groups[forced_group] = "train"
    group_splits = {}
    label_splits = {}
    for row in rows:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
        label_splits.setdefault(row["text"], set()).add(row["split"])
    if len(rows) != count or label_errors:
        raise RuntimeError(f"Dataset generation rejected {count - len(rows)} samples: {label_errors[:5]}")

    with (output / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "labels.txt").open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(f"{row['image']}\t{row['text']}\n")
    charset = sorted({character for row in rows for character in row["text"]}, key=ord)
    (output / "charset.txt").write_text("".join(f"{character}\n" for character in charset), encoding="utf-8")

    categories = Counter(row["category"] for row in rows)
    layouts = Counter(row["layout"] for row in rows)
    difficulties = Counter(row["difficulty"] for row in rows)
    splits = Counter(row["split"] for row in rows)
    info = {
        "generator": "scripts/generate_mixed_ocr_dataset.py",
        "generator_version": 2,
        "seed": args.seed,
        "total_samples": len(rows),
        "training_samples": splits["train"],
        "validation_samples": splits["val"],
        "unique_registration_strings": len(seen_labels),
        "unique_groups": len(seen_groups),
        "charset_size": len(charset),
        "charset": "".join(charset),
        "minimum_label_length": min(map(len, seen_labels)),
        "maximum_label_length": max(map(len, seen_labels)),
        "layout_counts": dict(layouts),
        "category_counts": dict(categories),
        "difficulty_counts": dict(difficulties),
        "background_counts": dict(Counter(row["background"] for row in rows)),
        "fonts": [name for _, _, name in fonts],
        "font_policy": "complete Latin + Devanagari cmap coverage required; no Pillow fallback",
        "validation_fraction_requested": args.val_fraction,
        "split_strategy": "sha256(group_id); all related variants remain in one split",
        "unicode_normalization": "NFC",
        "two_line_training_preprocess": "horizontal_stitch_v1",
        "decorative_header_training_preprocess": "registration_row_crop_v1",
    }
    write_json(output / "dataset_info.json", info)
    derived_charset = {character for row in rows for character in row["text"]}
    declared_charset = set(charset)
    group_overlap = sorted(group for group, split_values in group_splits.items() if split_values == {"train", "val"})
    label_overlap = sorted(label for label, split_values in label_splits.items() if split_values == {"train", "val"})
    integrity = {
        "status": "PASS" if not label_errors and derived_charset == declared_charset and not group_overlap and not label_overlap else "FAIL",
        "checked_samples": len(rows),
        "missing_images": [],
        "malformed_labels": label_errors,
        "duplicate_charset_entries": len(charset) != len(set(charset)),
        "charset_matches_labels": derived_charset == declared_charset,
        "train_validation_group_overlap": group_overlap,
        "train_validation_label_overlap": label_overlap,
        "zero_width_characters": sorted(derived_charset & ZERO_WIDTH),
        "control_characters": sorted(character for character in derived_charset if unicodedata.category(character).startswith("C")),
    }
    write_json(output / "integrity_report.json", integrity)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Integrity: {integrity['status']}")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/mixed_ocr_dataset")
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--smoke", action="store_true", help="Generate a small diagnostic dataset (use with --count)")
    args = parser.parse_args()
    if args.count < 10:
        parser.error("--count must be at least 10")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    return args


if __name__ == "__main__":
    generate(parse_args())
