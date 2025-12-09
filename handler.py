import base64
import io
import os
from typing import Any, Dict

import runpod
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"

print("Device:", DEVICE, "dtype:", DTYPE)

# Load SDXL Turbo (supports img2img)
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=DTYPE,
    use_safetensors=True
).to(DEVICE)

# Load LoRA
lora_path = os.path.join(LORA_DIR, LORA_NAME)
if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA missing: {lora_path}")

print(f"Loading LoRA from {lora_path}")
pipe.load_lora_weights(LORA_DIR, weight_name=LORA_NAME)
pipe.fuse_lora(lora_scale=1.0)

print("LoRA loaded & fused.")

def decode_image(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def handler(event: Dict[str, Any]):
    inp = event.get("input", {}) or {}

    prompt = inp.get("prompt", "")
    strength = float(inp.get("strength", 0.35))
    steps = int(inp.get("steps", 30))
    image_b64 = inp.get("image")

    if not image_b64:
        return {"error": "Missing base64 image"}

    try:
        img = decode_image(image_b64)
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    with torch.no_grad(), torch.autocast(DEVICE):
        out = pipe(
            prompt=prompt,
            image=img,
            strength=strength,
            num_inference_steps=steps
        ).images[0]

    return {"refined_image": encode_image(out)}

runpod.serverless.start({"handler": handler})
