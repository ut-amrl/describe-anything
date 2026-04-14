# DAM Server API Guide

This document describes how to integrate with the Describe Anything Model (DAM) HTTP server (`dam_server.py`). It is intended for both human developers and AI agents calling the API programmatically.

---

## Table of Contents

1. [Overview](#overview)
2. [Starting the Server](#starting-the-server)
3. [Image Format](#image-format)
4. [Endpoints](#endpoints)
   - [POST /chat/completions](#post-chatcompletions)
   - [POST /batch/chat/completions](#post-batchchatcompletions)
5. [Python Integration Examples](#python-integration-examples)
6. [curl Examples](#curl-examples)
7. [Error Handling](#error-handling)
8. [Configuration Reference](#configuration-reference)

---

## Overview

The DAM server exposes an OpenAI-compatible HTTP API for describing masked regions in images. Given an RGBA image (RGB = the scene, alpha channel = the region mask) and a natural-language prompt, the model returns a text description of the masked region.

Two endpoints are available:

| Endpoint | Purpose |
|---|---|
| `POST /chat/completions` | Describe a single masked region (supports streaming) |
| `POST /batch/chat/completions` | Describe multiple masked regions in one request |

---

## Starting the Server

### Directly with Python

```bash
python dam_server.py \
  --model-path nvidia/DAM-3B \
  --conv-mode v1 \
  --prompt-mode focal_prompt \
  --temperature 0.2 \
  --top_p 0.9 \
  --num_beams 1 \
  --max_new_tokens 512 \
  --host 0.0.0.0 \
  --port 8000
```

### Via the run script (Docker / container)

```bash
# Default port is 9014
DAM_API_PORT=9014 bash docker/run_describe_anything_api.sh
```

Environment variables accepted by the run script:

| Variable | Default | Description |
|---|---|---|
| `DAM_API_PORT` | `9014` | Port to listen on |
| `DAM_API_HOST` | `0.0.0.0` | Host to bind |
| `DAM_APP_MODULE` | `dam_server:app` | Uvicorn app module |
| `DAM_PROJECT_ROOT` | `/workspace/project` | Path to repo root |
| `CUDA_VISIBLE_DEVICES` | *(unset)* | GPU selection (takes priority) |
| `DAM_API_CUDA_VISIBLE_DEVICES` | *(unset)* | Fallback GPU selection |

### Waiting for readiness

Use `docker/smoke_test_describe_anything_api.sh` to poll until the server is ready:

```bash
bash docker/smoke_test_describe_anything_api.sh --url http://localhost:9014 --timeout-seconds 120
```

The script exits 0 when the server responds with HTTP 200 or 422 (validation error — meaning the route is up).

---

## Image Format

**Every image sent to the API must be RGBA PNG or JPEG.**

- **RGB channels** — the scene / photograph.
- **Alpha channel** — the mask. Any pixel with alpha > 0 is considered part of the region of interest. Fully masked = alpha 255 everywhere; region mask = alpha 255 inside the region, 0 outside.

Images are passed as base64-encoded data URIs:

```
data:image/png;base64,<base64-encoded RGBA bytes>
```

### Creating an RGBA image in Python

```python
from PIL import Image
import numpy as np
import base64
from io import BytesIO

def make_rgba_data_uri(rgb_image: Image.Image, mask: np.ndarray) -> str:
    """
    rgb_image : PIL RGB image
    mask      : 2-D uint8 numpy array, same H×W as rgb_image.
                Pixels > 0 are the region of interest.
    Returns a data URI string suitable for the API.
    """
    rgba = np.zeros((*np.array(rgb_image).shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = np.array(rgb_image)
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 255

    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    pil_rgba.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
```

---

## Endpoints

### POST /chat/completions

Describe a single masked region. Follows the OpenAI chat completions shape.

#### Request body

```json
{
  "model": "describe_anything_model",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "<RGBA data URI>" }
        },
        {
          "type": "text",
          "text": "Describe this region in detail."
        }
      ]
    }
  ],
  "max_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.9,
  "num_beams": 1,
  "stream": false
}
```

**Field notes:**

| Field | Required | Description |
|---|---|---|
| `model` | Yes | Use `"describe_anything_model"` to bypass model-name validation, or the exact HuggingFace model ID (e.g. `"nvidia/DAM-3B"`). |
| `messages` | Yes | List of chat messages. The RGBA image and prompt text must both appear in the `"user"` message. |
| `stream` | No | Set `true` to receive a Server-Sent Events stream. |
| `max_tokens`, `temperature`, `top_p`, `num_beams` | No | Accepted but currently the server uses its startup CLI values; these fields are present for OpenAI-client compatibility. |

#### Non-streaming response

```json
{
  "id": "3f2a...",
  "object": "chat.completion",
  "created": 1712345678.0,
  "model": "describe_anything_model",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The masked region shows a golden retriever sitting on a wooden floor..."
      }
    }
  ]
}
```

#### Streaming response

Each SSE chunk:

```
data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"delta":{"content":[{"type":"text","text":"The "}]},"finish_reason":null}]}

data: {"choices":[{"finish_reason":"stop"}]}

data: [DONE]
```

---

### POST /batch/chat/completions

Describe multiple masked regions in a single HTTP round-trip. Each item in `requests` is a standard `ChatCompletionRequest`. Items are processed sequentially on the GPU; results are returned in the same order. Streaming is not supported for batch requests.

#### Request body

```json
{
  "model": "describe_anything_model",
  "requests": [
    {
      "model": "describe_anything_model",
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "image_url", "image_url": { "url": "<RGBA data URI 1>" } },
            { "type": "text", "text": "What is this object?" }
          ]
        }
      ]
    },
    {
      "model": "describe_anything_model",
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "image_url", "image_url": { "url": "<RGBA data URI 2>" } },
            { "type": "text", "text": "Describe the material and color." }
          ]
        }
      ]
    }
  ]
}
```

#### Response

```json
{
  "object": "batch",
  "results": [
    {
      "id": "a1b2...",
      "object": "chat.completion",
      "created": 1712345678.0,
      "model": "describe_anything_model",
      "choices": [
        { "message": { "role": "assistant", "content": "A ceramic coffee mug..." } }
      ]
    },
    {
      "id": "c3d4...",
      "object": "chat.completion",
      "created": 1712345679.0,
      "model": "describe_anything_model",
      "choices": [
        { "message": { "role": "assistant", "content": "The surface appears to be brushed aluminum..." } }
      ]
    }
  ]
}
```

If one item fails, its slot contains `{"error": "<message>"}` and processing continues for the remaining items. A top-level HTTP 500 is only returned if the batch itself fails before any items are processed (e.g. invalid model name).

---

## Python Integration Examples

### Single request

```python
import requests
from PIL import Image
import numpy as np
import base64
from io import BytesIO

SERVER = "http://localhost:9014"

def make_rgba_data_uri(rgb_image: Image.Image, mask: np.ndarray) -> str:
    rgba = np.zeros((*np.array(rgb_image).shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = np.array(rgb_image)
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 255
    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# Load your image and mask
rgb = Image.open("photo.jpg").convert("RGB")
mask = np.zeros((rgb.height, rgb.width), dtype=np.uint8)
mask[100:200, 150:300] = 255  # mark the region of interest

data_uri = make_rgba_data_uri(rgb, mask)

response = requests.post(f"{SERVER}/chat/completions", json={
    "model": "describe_anything_model",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": "Describe this region in detail."},
        ],
    }],
})
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

### Batch request (one image, multiple masks)

```python
import requests
from PIL import Image
import numpy as np
import base64
from io import BytesIO

SERVER = "http://localhost:9014"

def make_rgba_data_uri(rgb_image: Image.Image, mask: np.ndarray) -> str:
    rgba = np.zeros((*np.array(rgb_image).shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = np.array(rgb_image)
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 255
    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

rgb = Image.open("photo.jpg").convert("RGB")

masks = [
    np.zeros((rgb.height, rgb.width), dtype=np.uint8),
    np.zeros((rgb.height, rgb.width), dtype=np.uint8),
]
masks[0][50:150, 50:200] = 255   # region A
masks[1][200:350, 100:400] = 255  # region B

def make_item(data_uri, prompt):
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

batch_items = [
    make_item(make_rgba_data_uri(rgb, masks[0]), "What is this object?"),
    make_item(make_rgba_data_uri(rgb, masks[1]), "Describe the material and texture."),
]

response = requests.post(f"{SERVER}/batch/chat/completions", json={
    "model": "describe_anything_model",
    "requests": batch_items,
})
response.raise_for_status()

for i, result in enumerate(response.json()["results"]):
    if "error" in result:
        print(f"Item {i} failed: {result['error']}")
    else:
        print(f"Item {i}: {result['choices'][0]['message']['content']}")
```

---

## curl Examples

### Single request (non-streaming)

```bash
# Build a tiny 1×1 white RGBA PNG for testing
RGBA_URI="data:image/png;base64,$(python3 -c "
import base64, io
from PIL import Image
import numpy as np
buf = io.BytesIO()
Image.fromarray(np.ones((64,64,4), dtype='uint8')*255, 'RGBA').save(buf, 'PNG')
print(base64.b64encode(buf.getvalue()).decode(), end='')
")"

curl -s -X POST http://localhost:9014/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"describe_anything_model\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"$RGBA_URI\"}},
        {\"type\": \"text\", \"text\": \"Describe this region.\"}
      ]
    }]
  }" | python3 -m json.tool
```

### Streaming

```bash
curl -N -X POST http://localhost:9014/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"describe_anything_model","stream":true,"messages":[...]}'
```

---

## Error Handling

| HTTP status | Meaning |
|---|---|
| `200` | Success |
| `422` | Request validation error (malformed JSON or missing required fields) |
| `500` | Server-side error (model inference failed, bad image format, etc.) |

A `500` body looks like:
```json
{ "error": "<human-readable message>" }
```

For batch requests, per-item errors are embedded in the `results` array and do **not** cause a top-level `500`:
```json
{
  "object": "batch",
  "results": [
    { "error": "No image with mask found in input messages." },
    { "id": "...", "object": "chat.completion", ... }
  ]
}
```

**Common error causes:**

- Image is not RGBA — convert to RGBA before sending.
- Alpha channel is all zeros — the mask is empty; no region is selected.
- More than one image in a non-video request — only one RGBA image per item is supported in standard (non-joint-checkpoint) mode.
- `DEFAULT_IMAGE_TOKEN` (`<image>`) appears in the middle of the query text — the server strips leading `<image>` tokens but will reject them elsewhere.

---

## Configuration Reference

Generation parameters are set at server startup and apply to all requests:

| CLI flag | Default | Description |
|---|---|---|
| `--model-path` | `nvidia/DAM-3B` | HuggingFace model ID or local path |
| `--conv-mode` | `v1` | Conversation template (`v1`, `llama_3`, etc.) |
| `--prompt-mode` | `focal_prompt` | Crop strategy. Currently only `focal_prompt` is supported (maps to `full+focal_crop` internally) |
| `--temperature` | `0.2` | Sampling temperature (0 = greedy) |
| `--top_p` | `0.9` | Nucleus sampling threshold |
| `--num_beams` | `1` | Beam search width |
| `--max_new_tokens` | `512` | Maximum tokens to generate |
| `--workers` | `1` | Number of Uvicorn worker processes |
| `--image_video_joint_checkpoint` | *(flag)* | Set when using a joint image+video checkpoint (`nvidia/DAM-3B-Video`) |

Environment variable overrides (server startup only):

| Variable | CLI equivalent |
|---|---|
| `DAM_HOST` | `--host` |
| `DAM_PORT` | `--port` |
| `DAM_MODEL_PATH` | `--model-path` |
| `DAM_CONV_MODE` | `--conv-mode` |
| `DAM_WORKERS` | `--workers` |
