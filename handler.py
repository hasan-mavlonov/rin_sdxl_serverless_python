import base64
import io
import os
from typing import Any, Dict

import runpod
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# -------------------------------------------------------
# Device / dtype
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print("Device:", DEVICE, "dtype:", DTYPE)

# -------------------------------------------------------
# Load SDXL Base (correct model for LoRA)
# -------------------------------------------------------
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True
).to(DEVICE)

print("Loaded SDXL Base img2img pipeline.")


# -------------------------------------------------------
# Optional LoRA loading (if you want later)
# -------------------------------------------------------
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"
ADAPTER_NAME = "rin_adapter"

lora_path = os.path.join(LORA_DIR, LORA_NAME)
HAS_LORA = os.path.exists(lora_path)

if HAS_LORA:
    print(f"Loading LoRA adapter from {lora_path}")
    pipe.load_lora_weights(
        LORA_DIR,
        weight_name=LORA_NAME,
        adapter_name=ADAPTER_NAME
    )
    pipe.set_adapters([ADAPTER_NAME], adapter_weights=[1.0])
    print("LoRA loaded.")
else:
    print("WARNING: No LoRA found, running without.")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
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


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# -------------------------------------------------------
# MAIN HANDLER
# -------------------------------------------------------
def handler(event: Dict[str, Any]):
    """
    Expected input:
    {
      "input": {
        "image": "<base64>",
        "prompt": "...",
        "strength": 0.0-1.0,
        "steps": int,
        "guidance_scale": float,
        "lora_scale": float,
        "seed": int
      }
    }
    """
    inp = event.get("input") or {}

    prompt = str(inp.get("prompt", ""))

    # REQUIRED
    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing base64 image"}

    # SAFE DEFAULTS (fixing black image problem)
    strength = float(inp.get("strength", 0.08))  # SAFE RANGE
    strength = _clamp(strength, 0.02, 0.15)

    steps = int(inp.get("steps", 12))
    steps = max(5, min(steps, 20))

    guidance_scale = float(inp.get("guidance_scale", 1.5))
    guidance_scale = _clamp(guidance_scale, 0.8, 3.0)

    lora_scale = float(inp.get("lora_scale", 1.0))
    lora_scale = _clamp(lora_scale, 0.0, 2.0)

    seed = inp.get("seed")

    # Decode input image
    try:
        img = decode_image(image_b64)
    except Exception as exc:
        return {"error": f"Failed to decode input image: {exc}"}

    # Init seed
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        except Exception:
            pass

    # Activate LoRA (if exists)
    if HAS_LORA:
        pipe.set_adapters([ADAPTER_NAME], adapter_weights=[lora_scale])

    # -------------------------------------------------------
    # Inference (NO MORE NaNs / black output)
    # -------------------------------------------------------
    try:
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16
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
                "used_lora": HAS_LORA
            }
        }

    except Exception as exc:
        return {"error": f"Pipeline failed: {exc}"}


runpod.serverless.start({"handler": handler})
