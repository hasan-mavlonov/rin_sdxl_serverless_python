import base64
import io
import os
from typing import Any, Dict

import runpod
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# -----------------------------
# Global config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Where the LoRA file lives (works locally and in Docker)
# In Docker: /app/lora/rinxl_lora.safetensors
# Locally:   ./lora/rinxl_lora.safetensors
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_WEIGHT_NAME = "rinxl_lora.safetensors"

print(f"Using device={DEVICE}, dtype={DTYPE}")
print(f"LoRA dir: {LORA_DIR}, weight: {LORA_WEIGHT_NAME}")

# -----------------------------
# Load base SDXL Turbo pipeline
# -----------------------------
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=DTYPE,
).to(DEVICE)

# -----------------------------
# Load LoRA and fuse into pipeline
# -----------------------------
lora_path = os.path.join(LORA_DIR, LORA_WEIGHT_NAME)

if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA file not found at {lora_path}")

print(f"Loading LoRA from {lora_path} ...")
# This loads the LoRA into the SDXL Turbo pipeline
pipe.load_lora_weights(
    LORA_DIR,
    weight_name=LORA_WEIGHT_NAME,
)

# Optional: fuse LoRA into UNet for speed (no extra runtime overhead)
# You can tune lora_scale (0.0–1.0) to control strength
pipe.fuse_lora(lora_scale=1.0)
print("LoRA loaded and fused.")


# -----------------------------
# Helpers for base64 <-> PIL
# -----------------------------
def decode_image(image_b64: str) -> Image.Image:
    """Decode a base64 string to a PIL image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(image: Image.Image) -> str:
    """Encode a PIL image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# -----------------------------
# RunPod handler
# -----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_data = event.get("input", {}) or {}

    prompt = input_data.get("prompt", "")
    strength = float(input_data.get("strength", 0.55))
    # SDXL Turbo usually works best with 1–6 steps; 4 is common
    steps = int(input_data.get("steps", 4))
    image_b64 = input_data.get("image")

    if not image_b64:
        return {"error": "Missing required field: image"}

    try:
        init_image = decode_image(image_b64)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to decode image: {exc}"}

    # Inference
    with torch.no_grad(), torch.autocast("cuda" if DEVICE == "cuda" else "cpu"):
        out_image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
        ).images[0]

    refined_b64 = encode_image(out_image)

    return {"refined_image": refined_b64}


runpod.serverless.start({"handler": handler})
