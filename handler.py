import base64
import io
import os
from typing import Any, Dict, Optional

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
# Load SDXL Base
# -------------------------------------------------------
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,  # load weights in fp16
    use_safetensors=True,
).to(DEVICE)

print("Loaded SDXL Base img2img pipeline.")

# -------------------------------------------------------
# 🔥 CRITICAL FIX: FP32 UNet + VAE (prevents NaNs)
# -------------------------------------------------------
pipe.text_encoder.to(torch.float16)
pipe.text_encoder_2.to(torch.float16)

pipe.unet.to(torch.float32)
pipe.vae.to(torch.float32)

print("SDXL stabilized: UNet + VAE running in FP32")

# -------------------------------------------------------
# Optional LoRA loading
# -------------------------------------------------------
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"
ADAPTER_NAME = "rin_adapter"

lora_path = os.path.join(LORA_DIR, LORA_NAME)
HAS_LORA = False

if os.path.exists(lora_path):
    try:
        print(f"Loading LoRA adapter from {lora_path}")
        pipe.load_lora_weights(
            LORA_DIR,
            weight_name=LORA_NAME,
            adapter_name=ADAPTER_NAME,
        )
        HAS_LORA = True
        print("LoRA loaded.")
    except Exception as exc:
        HAS_LORA = False
        print(f"LoRA failed to load: {exc}")

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


def _fix_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    w = max(512, (w // 8) * 8)
    h = max(512, (h // 8) * 8)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img


def _safe_generator(seed: Optional[int]):
    if seed is None:
        return None
    try:
        return torch.Generator(device=DEVICE).manual_seed(int(seed))
    except Exception:
        return None


# -------------------------------------------------------
# MAIN HANDLER
# -------------------------------------------------------
def handler(event: Dict[str, Any]):
    inp = event.get("input") or {}

    prompt = str(inp.get("prompt", ""))

    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing base64 image"}

    strength = _clamp(float(inp.get("strength", 0.3)), 0.25, 0.6)
    steps = max(10, min(int(inp.get("steps", 16)), 40))
    guidance_scale = _clamp(float(inp.get("guidance_scale", 3.0)), 1.0, 7.5)

    lora_scale = _clamp(float(inp.get("lora_scale", 0.85)), 0.0, 1.25)
    seed = inp.get("seed")

    try:
        img = decode_image(image_b64)
    except Exception as exc:
        return {"error": f"Image decode failed: {exc}"}

    img = _fix_size(img)
    generator = _safe_generator(seed)

    # Enable / disable LoRA safely
    if HAS_LORA and lora_scale > 0:
        pipe.set_adapters([ADAPTER_NAME], adapter_weights=[lora_scale])
    else:
        try:
            pipe.set_adapters([], [])
        except Exception:
            pass

    try:
        with torch.inference_mode():  # ❌ NO autocast
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
                "lora_scale": lora_scale if HAS_LORA else 0.0,
                "seed": seed,
                "device": DEVICE,
                "used_lora": HAS_LORA and lora_scale > 0,
            },
        }

    except Exception as exc:
        return {"error": f"Pipeline failed: {exc}"}


runpod.serverless.start({"handler": handler})
