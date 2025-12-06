# lora_pipeline.py

import os
from pathlib import Path
from io import BytesIO

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import requests


# ---------- Config ----------

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# We’ll download the LoRA from this URL once.
# Replace this with your actual direct URL (e.g. GitHub raw, Supabase, etc.)
DEFAULT_LORA_URL = "https://YOUR_HOST/rinxl_lora.safetensors"

# Directory relative to this file
ROOT_DIR = Path(__file__).resolve().parent
LORA_DIR = ROOT_DIR / "lora"
LORA_PATH = LORA_DIR / "rinxl_lora.safetensors"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_pipe = None  # global cached pipeline


# ---------- Helpers ----------

def _ensure_lora_exists():
    """Download the LoRA file once if it's not on disk yet."""
    if LORA_PATH.exists():
        return

    LORA_DIR.mkdir(parents=True, exist_ok=True)

    lora_url = os.getenv("RIN_LORA_URL", DEFAULT_LORA_URL)
    if not lora_url or "YOUR_HOST" in lora_url:
        raise RuntimeError(
            "LoRA URL is not configured. "
            "Set RIN_LORA_URL env or edit DEFAULT_LORA_URL in lora_pipeline.py."
        )

    print(f"[SDXL-LoRA] Downloading LoRA from {lora_url} ...")
    resp = requests.get(lora_url, timeout=60)
    resp.raise_for_status()

    with open(LORA_PATH, "wb") as f:
        f.write(resp.content)

    print(f"[SDXL-LoRA] LoRA saved to {LORA_PATH}")


def _get_pipeline():
    """Create and cache the SDXL img2img pipeline with LoRA loaded."""
    global _pipe
    if _pipe is not None:
        return _pipe

    _ensure_lora_exists()

    print("[SDXL-LoRA] Loading base SDXL model...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(DEVICE)

    print(f"[SDXL-LoRA] Loading LoRA from {LORA_PATH} ...")
    pipe.load_lora_weights(str(LORA_PATH))

    # Optional speed/VRAM tweaks:
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention = getattr(
        pipe, "enable_xformers_memory_efficient_attention", lambda: None
    )

    _pipe = pipe
    print("[SDXL-LoRA] Pipeline ready.")
    return _pipe


# ---------- Public API ----------

def refine_sdxl(
    image_bytes: bytes,
    prompt: str,
    strength: float = 0.55,
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
):
    """
    Refine an input image with SDXL + LoRA using img2img.
    Returns PNG bytes.
    """
    pipe = _get_pipeline()

    init_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    buf = BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
