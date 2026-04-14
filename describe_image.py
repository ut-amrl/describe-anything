#!/usr/bin/env python3
"""
describe_image.py — Run DAM descriptions on segmented regions of an image.

Two input modes:
  --npz PATH    Pre-computed segmentation_results.npz (contains masks, ids,
                scores, and optionally the source imagePath).
  --sam3        Call the SAM3 service interactively to segment the image first.

Required when using --sam3, or when imagePath is absent from the NPZ:
  --image PATH

Output: a JSON file mapping instance_id -> description string.

Examples
--------
# Describe all regions from a pre-computed NPZ:
  python describe_image.py \\
      --npz /path/to/segmentation_results.npz \\
      --output descriptions.json

# Override the image path stored in the NPZ:
  python describe_image.py \\
      --npz /path/to/segmentation_results.npz \\
      --image /path/to/image.jpg

# Segment a fresh image with SAM3 (defaults to a center-point click):
  python describe_image.py --sam3 --image /path/to/image.jpg

# SAM3 with explicit click points:
  python describe_image.py --sam3 --image /path/to/image.jpg \\
      --points "320,240 100,80"

# SAM3 with a bounding-box prompt:
  python describe_image.py --sam3 --image /path/to/image.jpg \\
      --box 50,50,600,400
"""

import argparse
import base64
import colorsys
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_rgb(image_path: str) -> np.ndarray:
    """Load an image as (H, W, 3) uint8 RGB array."""
    return np.array(Image.open(image_path).convert("RGB"))


def make_rgba_data_uri(rgb: np.ndarray, mask: np.ndarray) -> str:
    """
    Compose an RGBA PNG data URI from an RGB image and a binary mask.

    rgb:  (H, W, 3) uint8
    mask: (H, W) bool or uint8 — nonzero pixels are the region of interest
    """
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_image_b64(rgb: np.ndarray) -> str:
    """Base64-encode an RGB numpy array as a PNG (for the SAM3 API)."""
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_mask_b64(mask_b64: str) -> np.ndarray:
    """Decode a base64 PNG mask returned by SAM3 → (H, W) bool array."""
    data = base64.b64decode(mask_b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img) > 0


def print_mask_summary(ids, masks, scores, boxes=None) -> None:
    """Print a human-readable per-entry table of mask metadata."""
    from collections import Counter
    id_counts = Counter(ids.tolist())
    dup_ids = {k for k, v in id_counts.items() if v > 1}

    header = f"{'idx':>4}  {'id':>5}  {'score':>6}  {'area':>7}  {'med_y':>6}  {'med_x':>6}  {'h':>5}  {'w':>5}"
    if boxes is not None:
        header += "  box (x0,y0,x1,y1)"
    print(header)
    print("-" * len(header))

    for i in range(len(ids)):
        m = masks[i].astype(bool)
        ys, xs = np.where(m)
        area = int(m.sum())
        if area > 0:
            med_y, med_x = int(np.median(ys)), int(np.median(xs))
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
        else:
            med_y, med_x, h, w = -1, -1, 0, 0
        sc = float(scores[i]) if scores is not None and scores.ndim > 0 else float("nan")
        row = f"{i:>4}  {ids[i]:>5}  {sc:>6.3f}  {area:>7}  {med_y:>6}  {med_x:>6}  {h:>5}  {w:>5}"
        if boxes is not None:
            b = boxes[i]
            row += f"  [{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]"
        if ids[i] in dup_ids:
            row += "  *"
        print(row)

    unique_ids = sorted(id_counts.keys())
    print(f"\n{len(ids)} entries, {len(unique_ids)} unique id(s): {unique_ids}")
    if dup_ids:
        print(f"WARNING: {len(dup_ids)} id(s) appear more than once (* marked): {sorted(dup_ids)}")
        print("         The output JSON will only keep one description per id.")


def save_debug_image(
    rgb: np.ndarray,
    mask_list: list,
    labels: list,
    out_dir: str,
    alpha: float = 0.45,
) -> None:
    """Save one debug PNG per mask into out_dir.

    Each image shows the source photo with a single mask highlighted in a
    semi-transparent yellow-green overlay and its label drawn at the centroid.

    rgb:       (H, W, 3) uint8 source image
    mask_list: list of (H, W) bool arrays
    labels:    list of labels (str/int) — same order as mask_list
    out_dir:   directory where per-mask PNGs are written ({label}.png)
    alpha:     opacity of the color overlay
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    highlight = np.array([255, 220, 0], dtype=np.float32)  # yellow

    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for mask, label in zip(mask_list, labels):
        canvas = rgb.astype(np.float32).copy()
        where = np.asarray(mask) > 0
        canvas[where] = (1 - alpha) * canvas[where] + alpha * highlight

        img = Image.fromarray(canvas.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        ys, xs = np.where(where)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            text = str(label)
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                           (0, -1), (0, 1), (-1, 0), (1, 0)]:
                draw.text((cx + dx, cy + dy), text, fill=(0, 0, 0), font=font)
            draw.text((cx, cy), text, fill=(255, 255, 255), font=font)

        out_path = out_dir_path / f"{label}.png"
        img.save(out_path)

    print(f"Debug images ({len(mask_list)}) saved to {out_dir}/")


def save_per_id_debug_images(
    rgb: np.ndarray,
    ids,
    masks,
    scores,
    out_dir: str,
    alpha: float = 0.45,
) -> None:
    """Save one debug PNG per unique instance ID.

    Each image shows all mask proposals for that ID overlaid in distinct colors,
    labeled with their array index and score.

    ids:     (N,) array of instance IDs (may have duplicates)
    masks:   (N, H, W) bool array
    scores:  (N,) float array or None
    out_dir: directory where per-ID PNGs are written
    """
    from collections import defaultdict

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    groups: dict = defaultdict(list)  # id -> list of array indices
    for i, iid in enumerate(ids.tolist()):
        groups[iid].append(i)

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for iid, indices in sorted(groups.items()):
        n = len(indices)
        canvas = rgb.astype(np.float32).copy()

        colors = []
        h_val = 0.0
        for _ in range(n):
            h_val = (h_val + 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h_val, 0.75, 0.95)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))

        for idx, color in zip(indices, colors):
            where = masks[idx].astype(bool)
            canvas[where] = (
                (1 - alpha) * canvas[where]
                + alpha * np.array(color, dtype=np.float32)
            )

        img = Image.fromarray(canvas.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        for idx, color in zip(indices, colors):
            m = masks[idx].astype(bool)
            ys, xs = np.where(m)
            if len(ys) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            sc = float(scores[idx]) if scores is not None and scores.ndim > 0 else float("nan")
            text = f"{idx}  {sc:.2f}"
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                           (0, -1), (0, 1), (-1, 0), (1, 0)]:
                draw.text((cx + dx, cy + dy), text, fill=(0, 0, 0), font=font)
            draw.text((cx, cy), text, fill=(255, 255, 255), font=font)

        out_path = out_dir_path / f"id_{iid}.png"
        img.save(out_path)
        print(f"  id={iid}: {n} proposal(s) -> {out_path}")

    print(f"Per-ID debug images saved to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# SAM3 client
# ──────────────────────────────────────────────────────────────────────────────

def segment_with_sam3(
    image: np.ndarray,
    sam3_url: str,
    points: list,
    box_xyxy: Optional[list],
    multimask_output: bool = True,
    timeout: int = 120,
) -> list:
    """
    Call POST /segment-image on the SAM3 service.

    Returns a list of dicts: {"mask": ndarray (H,W) bool, "score": float}.
    """
    payload = {
        "image_b64": encode_image_b64(image),
        "points": points,
        "box_xyxy": box_xyxy,
        "multimask_output": multimask_output,
    }
    url = f"{sam3_url.rstrip('/')}/segment-image"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR: SAM3 request to {url} failed: {exc}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    return [
        {"mask": decode_mask_b64(c["mask_b64"]), "score": float(c.get("score", 0.0))}
        for c in data.get("candidates", [])
    ]


# ──────────────────────────────────────────────────────────────────────────────
# DAM client
# ──────────────────────────────────────────────────────────────────────────────

def describe_with_dam_batch(
    rgb: np.ndarray,
    masks: list,
    dam_url: str,
    prompt: str,
    chunk_size: int = 32,
    timeout: int = 300,
) -> list:
    """
    Call POST /batch/chat/completions on the DAM service.

    Splits ``masks`` into sequential chunks of at most ``chunk_size`` to stay
    within the server's DAM_MAX_BATCH_SIZE limit.

    Returns a list of description strings in the same order as masks.
    Per-item inference errors are returned as strings prefixed "ERROR: ".
    """
    def make_item(data_uri):
        return {
            "model": "describe_anything_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }

    url = f"{dam_url.rstrip('/')}/batch/chat/completions"
    descriptions = []

    for chunk_start in range(0, len(masks), chunk_size):
        chunk_masks = masks[chunk_start: chunk_start + chunk_size]
        items = [make_item(make_rgba_data_uri(rgb, m)) for m in chunk_masks]
        payload = {"model": "describe_anything_model", "requests": items}

        if len(masks) > chunk_size:
            chunk_end = min(chunk_start + chunk_size, len(masks))
            print(f"  chunk {chunk_start + 1}–{chunk_end} of {len(masks)} ...")

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"ERROR: DAM request to {url} failed: {exc}", file=sys.stderr)
            sys.exit(1)

        for r in resp.json().get("results", []):
            if "error" in r:
                descriptions.append(f"ERROR: {r['error']}")
            else:
                descriptions.append(r["choices"][0]["message"]["content"])

    return descriptions


def describe_with_dam_serial(
    rgb: np.ndarray,
    masks: list,
    dam_url: str,
    prompt: str,
    timeout: int = 300,
) -> list:
    """
    Call POST /chat/completions once per mask on the DAM service.

    Returns a list of description strings in the same order as masks.
    Per-item inference errors are returned as strings prefixed "ERROR: ".
    """
    url = f"{dam_url.rstrip('/')}/chat/completions"
    payload_template = {
        "model": "describe_anything_model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": None}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    descriptions = []

    for i, mask in enumerate(masks):
        payload_template["messages"][0]["content"][0]["image_url"]["url"] = (
            make_rgba_data_uri(rgb, mask)
        )
        try:
            resp = requests.post(url, json=payload_template, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"ERROR: DAM request to {url} failed: {exc}", file=sys.stderr)
            sys.exit(1)

        data = resp.json()
        if "error" in data:
            descriptions.append(f"ERROR: {data['error']}")
        else:
            descriptions.append(data["choices"][0]["message"]["content"])

        print(f"  {i + 1}/{len(masks)}", end="\r", flush=True)

    print()  # clear the \r line
    return descriptions


# ──────────────────────────────────────────────────────────────────────────────
# NPZ loading
# ──────────────────────────────────────────────────────────────────────────────

def load_npz(npz_path: str):
    """
    Load a segmentation_results.npz produced by SegmentationResults.saveToFile().

    Returns (ids, masks, scores, image_path):
      ids:        (N,) int64 ndarray
      masks:      (N, H, W) bool ndarray
      scores:     (N,) float32 ndarray or None
      image_path: str or None
    """
    with np.load(npz_path, allow_pickle=True) as data:
        ids = data["ids"]
        masks = data["masks"].astype(bool)
        scores_raw = data["scores"]
        # scores is saved as a 0-d object array containing None when absent
        scores = None if scores_raw.ndim == 0 else scores_raw.astype(np.float32)
        image_path = data["imagePath"].item() if "imagePath" in data else None
    return ids, masks, scores, image_path


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--npz", metavar="PATH",
        help="Pre-computed segmentation_results.npz",
    )
    src.add_argument(
        "--sam3", action="store_true",
        help="Segment the image interactively via the SAM3 HTTP service",
    )

    parser.add_argument(
        "--image", metavar="PATH",
        help="Input image path. Required for --sam3. In --npz mode, overrides "
             "the imagePath stored inside the NPZ.",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output JSON file (default: descriptions.json in the same directory "
             "as the image, or the current directory if unknown)",
    )
    parser.add_argument(
        "--debug-image", metavar="DIR",
        help="Save one debug PNG per mask into DIR, each showing the mask "
             "highlighted in yellow and labeled with its output key.",
    )
    parser.add_argument(
        "--debug-per-id", metavar="DIR",
        help="Save one debug PNG per unique instance ID into DIR, showing all "
             "mask proposals for that ID overlaid with their scores.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this region in detail.",
        help="Prompt sent to DAM for every region (default: %(default)r)",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.0, metavar="F",
        help="Discard masks with score < F (default: %(default)s)",
    )
    parser.add_argument(
        "--dedup", action="store_true", default=False,
        help="When set, keep only the highest-scoring mask per unique instance ID "
             "and key output by ID. When unset (default), describe every mask and "
             "key output by array index.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print a per-entry summary of the loaded masks (id, score, area, "
             "median point, bounding-box size) before running DAM.",
    )
    parser.add_argument(
        "--dam-url", default="http://localhost:9014", metavar="URL",
        help="DAM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N",
        help="Max masks per DAM batch call; must not exceed the server's "
             "DAM_MAX_BATCH_SIZE (default: %(default)s)",
    )
    parser.add_argument(
        "--serial", action="store_true", default=False,
        help="Call /chat/completions once per mask instead of using the batch "
             "endpoint. Useful for comparing throughput.",
    )
    parser.add_argument(
        "--sam3-url", default="http://localhost:9012", metavar="URL",
        help="SAM3 server base URL (default: %(default)s)",
    )

    sam3_grp = parser.add_argument_group("SAM3 prompts (only used with --sam3)")
    pt_or_box = sam3_grp.add_mutually_exclusive_group()
    pt_or_box.add_argument(
        "--points", metavar="\"x1,y1 x2,y2 ...\"",
        help="Foreground click points as space-separated x,y pairs "
             "(e.g. \"320,240 100,80\"). Defaults to the image center if omitted.",
    )
    pt_or_box.add_argument(
        "--box", metavar="x1,y1,x2,y2",
        help="Bounding-box prompt in pixel coordinates (XYXY format).",
    )

    return parser


def resolve_npz_image_path(stored_path: str) -> str:
    """
    Translate a path stored inside an NPZ that was written from inside the
    container (where /workspace/data maps to the host data directory).
    """
    container_prefix = "/workspace/data"
    host_prefix = "/scratch/aaadkins/data/semantic-persistence-filter"
    if stored_path.startswith(container_prefix):
        return host_prefix + stored_path[len(container_prefix):]
    return stored_path


def resolve_output_path(args_output: Optional[str], image_path: Optional[str]) -> Path:
    if args_output:
        return Path(args_output)
    if image_path:
        return Path(image_path).parent / "descriptions.json"
    return Path("descriptions.json")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve masks and image ───────────────────────────────────────────────
    if args.npz:
        ids, masks, scores, stored_image_path = load_npz(args.npz)

        image_path = args.image or (resolve_npz_image_path(stored_image_path) if stored_image_path else None)
        if not image_path:
            parser.error(
                "The NPZ file does not contain an imagePath. "
                "Please supply --image PATH."
            )

        rgb = load_rgb(image_path)

        with np.load(args.npz, allow_pickle=True) as raw:
            boxes = raw["boxes"] if "boxes" in raw else None

        if args.verbose:
            print(f"\n=== NPZ contents: {args.npz} ===")
            print_mask_summary(ids, masks, scores, boxes=boxes)
            print()

        if args.score_thresh > 0.0 and scores is not None:
            keep = scores >= args.score_thresh
            ids = ids[keep]
            masks = masks[keep]
            scores = scores[keep]
            if boxes is not None:
                boxes = boxes[keep]

        if args.dedup:
            # Keep only the highest-scoring mask per unique instance ID.
            best: dict = {}  # id -> (score, array_index)
            for i, iid in enumerate(ids.tolist()):
                score = float(scores[i]) if scores is not None else 0.0
                if iid not in best or score > best[iid][0]:
                    best[iid] = (score, i)
            instance_ids = list(best.keys())
            mask_list = [masks[best[iid][1]] for iid in instance_ids]
            if len(ids) > len(instance_ids):
                print(
                    f"  --dedup: kept {len(instance_ids)} best-scoring mask(s) "
                    f"from {len(ids)} proposals "
                    f"({len(ids) - len(instance_ids)} lower-scoring duplicate(s) dropped)."
                )
        else:
            # Describe every mask; use array index as the output key.
            instance_ids = list(range(len(ids)))
            mask_list = [masks[i] for i in instance_ids]

    else:  # --sam3
        if not args.image:
            parser.error("--image is required in --sam3 mode")

        image_path = args.image
        rgb = load_rgb(image_path)
        h, w = rgb.shape[:2]

        # Build SAM3 click / box prompt
        box_xyxy = None
        if args.box:
            box_xyxy = [float(v) for v in args.box.split(",")]
            points = []
        elif args.points:
            points = []
            for token in args.points.split():
                x_str, y_str = token.split(",")
                points.append({"x": float(x_str), "y": float(y_str), "label": 1})
        else:
            # Default: single center-point foreground click
            points = [{"x": w / 2, "y": h / 2, "label": 1}]
            print(f"No --points or --box given; using image center ({w // 2}, {h // 2})")

        candidates = segment_with_sam3(
            image=rgb,
            sam3_url=args.sam3_url,
            points=points,
            box_xyxy=box_xyxy,
        )

        if args.score_thresh > 0.0:
            candidates = [c for c in candidates if c["score"] >= args.score_thresh]

        if not candidates:
            print(
                "ERROR: SAM3 returned no mask candidates above the score threshold.",
                file=sys.stderr,
            )
            return 1

        instance_ids = list(range(len(candidates)))
        mask_list = [c["mask"] for c in candidates]

    if not mask_list:
        print("No masks to describe.", file=sys.stderr)
        return 1

    if args.debug_image:
        save_debug_image(rgb, mask_list, instance_ids, args.debug_image)

    if getattr(args, "debug_per_id", None) and args.npz:
        print(f"Saving per-ID debug images to {args.debug_per_id}/ ...")
        save_per_id_debug_images(rgb, ids, masks, scores, args.debug_per_id)

    mode = "serial" if args.serial else f"batch (chunk_size={args.batch_size})"
    print(f"Describing {len(mask_list)} mask(s) via DAM at {args.dam_url} [{mode}] ...")
    t0 = time.perf_counter()
    if args.serial:
        descriptions = describe_with_dam_serial(
            rgb=rgb,
            masks=mask_list,
            dam_url=args.dam_url,
            prompt=args.prompt,
        )
    else:
        descriptions = describe_with_dam_batch(
            rgb=rgb,
            masks=mask_list,
            dam_url=args.dam_url,
            prompt=args.prompt,
            chunk_size=args.batch_size,
        )
    elapsed = time.perf_counter() - t0
    n = len(descriptions)
    print(f"  Done: {elapsed:.2f}s total | {n} description(s) | {elapsed / n:.2f}s avg each")

    output = {str(iid): desc for iid, desc in zip(instance_ids, descriptions)}

    out_path = resolve_output_path(args.output, image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(output)} description(s) -> {out_path}")

    errors = [(iid, desc) for iid, desc in output.items() if desc.startswith("ERROR:")]
    if errors:
        print(f"WARNING: {len(errors)} item(s) failed:")
        for iid, msg in errors:
            print(f"  id={iid}: {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
