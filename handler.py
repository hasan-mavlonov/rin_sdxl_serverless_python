import base64
import io
import os
from typing import Any, Dict

import runpod
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print("Device:", DEVICE, "dtype:", DTYPE)

# --- SDXL BASE, NO LORA ---
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True,
).to(DEVICE)

print("Loaded SDXL Base img2img WITHOUT LoRA.")


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))

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


def handler(event: Dict[str, Any]):
    inp = event.get("input") or {}

    prompt: str = inp.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing base64 image in 'input.image'"}

    try:
        strength = float(inp.get("strength", 0.25))
    except Exception:
        strength = 0.25
    strength = _clamp(strength, 0.0, 1.0)

    try:
        steps = int(inp.get("steps", 20))
    except Exception:
        steps = 20
    steps = max(5, min(steps, 50))

    try:
        guidance_scale = float(inp.get("guidance_scale", 4.5))
    except Exception:
        guidance_scale = 4.5

    seed = inp.get("seed")

    try:
        img = decode_image(image_b64)
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        except Exception:
            generator = None

    try:
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
                "seed": seed,
                "device": DEVICE,
                "adapter_name": None,
            },
        }
    except Exception as e:
        return {"error": f"Pipeline failed: {e}"}


runpod.serverless.start({"handler": handler})
