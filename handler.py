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

# LoRA location
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"

print("Device:", DEVICE, "dtype:", DTYPE)

# -----------------------------
# Load SDXL Base (supports LoRA!)
# -----------------------------
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DEVICE == "cuda" else None
).to(DEVICE)


# -----------------------------
# Load LoRA
# -----------------------------
lora_path = os.path.join(LORA_DIR, LORA_NAME)
if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA not found at {lora_path}")

print(f"Loading LoRA weight from {lora_path}")

pipe.load_lora_weights(
    LORA_DIR,
    weight_name=LORA_NAME
)

# Fuse LoRA for speed (optional)
pipe.fuse_lora(lora_scale=1.0)

print("LoRA loaded and fused successfully.")


# -----------------------------
# Helpers
# -----------------------------
def decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# -----------------------------
# Inference handler
# -----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_data = event.get("input", {}) or {}

    prompt = input_data.get("prompt", "")
    strength = float(input_data.get("strength", 0.35))  # SDXL img2img usually 0.25–0.45
    steps = int(input_data.get("steps", 30))
    image_b64 = input_data.get("image")

    if not image_b64:
        return {"error": "Missing base64 image"}

    try:
        init_image = decode_image(image_b64)
    except Exception as exc:
        return {"error": f"Failed to decode image: {exc}"}

    with torch.no_grad(), torch.autocast(DEVICE):
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=3.5,  # SDXL sweet spot
        ).images[0]

    return {"refined_image": encode_image(result)}


runpod.serverless.start({"handler": handler})
