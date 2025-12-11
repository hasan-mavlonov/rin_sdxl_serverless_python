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
# LoRA paths & adapter name
# -------------------------------
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"
ADAPTER_NAME = "rinxl"

lora_path = os.path.join(LORA_DIR, LORA_NAME)
if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA missing: {lora_path}")

# -------------------------------
# Load SDXL Turbo + LoRA (dynamic adapters)
# -------------------------------
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=DTYPE,
    use_safetensors=True
).to(DEVICE)

print(f"Loading LoRA from {lora_path} with adapter '{ADAPTER_NAME}'")
# Load LoRA as an adapter (NOT fused) so we can control lora_scale at runtime.
pipe.load_lora_weights(
    LORA_DIR,
    weight_name=LORA_NAME,
    adapter_name=ADAPTER_NAME,
)

# Enable adapter with default scale 1.0
pipe.set_adapters([ADAPTER_NAME], adapter_weights=[1.0])
print("LoRA loaded as dynamic adapter.")


# -------------------------------
# Helpers
# -------------------------------
def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))

    # Fix for Gemini PNG with alpha → black artifacts
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg.convert("RGB")

    return img.convert("RGB")



def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# -------------------------------
# Handler
# -------------------------------
def handler(event: Dict[str, Any]):
    """
    Expected input shape:

    {
      "input": {
        "prompt": "text prompt",
        "image": "<base64 PNG/JPEG>",
        "strength": 0.35,          # optional, default 0.35
        "steps": 4,                # optional, default 4
        "guidance_scale": 0.0,     # optional, default 0.0 (Turbo-style)
        "lora_scale": 1.0,         # optional, default 1.0
        "seed": 123                # optional
      }
    }
    """
    inp = event.get("input") or {}

    prompt: str = inp.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing base64 image in 'input.image'"}

    # SDXL Turbo is designed for very few steps, low guidance.
    try:
        strength = float(inp.get("strength", 0.35))
    except Exception:
        strength = 0.35
    strength = _clamp(strength, 0.0, 1.0)

    try:
        steps = int(inp.get("steps", 4))
    except Exception:
        steps = 4
    steps = max(1, min(steps, 20))  # Turbo really doesn't need many steps

    try:
        guidance_scale = float(inp.get("guidance_scale", 0.0))
    except Exception:
        guidance_scale = 0.0

    try:
        lora_scale = float(inp.get("lora_scale", 1.0))
    except Exception:
        lora_scale = 1.0
    # Reasonable range for LoRA influence
    lora_scale = _clamp(lora_scale, 0.0, 2.0)

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
            generator = None

    # Inference
    try:
        # Set LoRA adapter scale dynamically per request.
        pipe.set_adapters([ADAPTER_NAME], adapter_weights=[lora_scale])

        with torch.inference_mode(), torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16,
        ):
            result = pipe(
                prompt=prompt,
                image=img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
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
                "adapter_name": ADAPTER_NAME,
            },
        }
    except Exception as e:
        # Make debugging easier from your Python client
        return {"error": f"Pipeline failed: {e}"}


runpod.serverless.start({"handler": handler})
