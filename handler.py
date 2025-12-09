import base64
import io
import os
from typing import Any, Dict

import runpod
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# -------------------------------
# Device & dtypes
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# For GPU: fp16; for CPU: stay in fp32 and use bfloat16 for autocast
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Device:", DEVICE, "dtype:", DTYPE)

# -------------------------------
# LoRA paths
# -------------------------------
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"

lora_path = os.path.join(LORA_DIR, LORA_NAME)
if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA missing: {lora_path}")

# -------------------------------
# Load SDXL Turbo + LoRA
# -------------------------------
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=DTYPE,
    use_safetensors=True
).to(DEVICE)

print(f"Loading LoRA from {lora_path}")
pipe.load_lora_weights(LORA_DIR, weight_name=LORA_NAME)
# fuse_lora makes it "baked in" (fast; cannot change scale later without reload)
pipe.fuse_lora(lora_scale=1.0)
print("LoRA loaded & fused.")


# -------------------------------
# Helpers
# -------------------------------
def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------
# Handler
# -------------------------------
def handler(event: Dict[str, Any]):
    inp = event.get("input") or {}

    prompt: str = inp.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing base64 image in 'input.image'"}

    # SDXL Turbo is designed for very few steps, low guidance.
    strength = float(inp.get("strength", 0.35))
    steps = int(inp.get("steps", 4))  # 4 is usually plenty for Turbo
    guidance_scale = float(inp.get("guidance_scale", 0.0))  # 0.0 recommended for Turbo
    lora_scale = float(inp.get("lora_scale", 1.0))
    seed = inp.get("seed")

    try:
        img = decode_image(image_b64)
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    # Optional seeding for reproducibility
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        except Exception:
            pass

    # Inference
    try:
        # If you want dynamic lora_scale, you would need set_adapters instead of fuse_lora.
        # With fuse_lora, this is fixed at load time (lora_scale above).
        with torch.inference_mode(), torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16
        ):
            result = pipe(
                prompt=prompt,
                image=img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

        out_img = result.images[0]
        return {
            "refined_image": encode_image(out_img),
            "meta": {
                "strength": strength,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "lora_scale": lora_scale,
                "seed": seed,
                "device": DEVICE,
            },
        }
    except Exception as e:
        # Make debugging easier from your Python client
        return {"error": f"Pipeline failed: {e}"}


runpod.serverless.start({"handler": handler})
