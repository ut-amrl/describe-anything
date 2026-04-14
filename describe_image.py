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
import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image


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
    timeout: int = 300,
) -> list:
    """
    Call POST /batch/chat/completions on the DAM service.

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

    items = [make_item(make_rgba_data_uri(rgb, m)) for m in masks]
    payload = {"model": "describe_anything_model", "requests": items}

    url = f"{dam_url.rstrip('/')}/batch/chat/completions"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR: DAM request to {url} failed: {exc}", file=sys.stderr)
        sys.exit(1)

    results = resp.json().get("results", [])
    descriptions = []
    for r in results:
        if "error" in r:
            descriptions.append(f"ERROR: {r['error']}")
        else:
            descriptions.append(r["choices"][0]["message"]["content"])
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
        "--prompt",
        default="Describe this region in detail.",
        help="Prompt sent to DAM for every region (default: %(default)r)",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.0, metavar="F",
        help="Discard masks with score < F (default: %(default)s)",
    )
    parser.add_argument(
        "--dam-url", default="http://localhost:9014", metavar="URL",
        help="DAM server base URL (default: %(default)s)",
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

        if args.score_thresh > 0.0 and scores is not None:
            keep = scores >= args.score_thresh
            ids = ids[keep]
            masks = masks[keep]
            scores = scores[keep]

        instance_ids = ids.tolist()
        mask_list = [masks[i] for i in range(len(ids))]

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

    print(f"Describing {len(mask_list)} mask(s) via DAM at {args.dam_url} ...")
    descriptions = describe_with_dam_batch(
        rgb=rgb,
        masks=mask_list,
        dam_url=args.dam_url,
        prompt=args.prompt,
    )

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
