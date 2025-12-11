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
    torch_dtype=DTYPE,
    use_safetensors=True,
).to(DEVICE)

print("Loaded SDXL Base img2img pipeline.")

# -------------------------------------------------------
# Optional LoRA loading
# -------------------------------------------------------
LORA_DIR = "/app/lora" if os.path.exists("/app/lora") else "lora"
LORA_NAME = "rinxl_lora.safetensors"
ADAPTER_NAME = "rin_adapter"

lora_path = os.path.join(LORA_DIR, LORA_NAME)
HAS_LORA = False


def _log(msg: str) -> None:
    """Lightweight logger for RunPod."""
    print(f"[handler] {msg}")


def _load_lora_adapter() -> bool:
    """Attempt to load the LoRA adapter; return True if active."""
    global HAS_LORA

    if not os.path.exists(lora_path):
        _log(f"LoRA not found at {lora_path}; proceeding without adapter.")
        return False

    try:
        _log(f"Loading LoRA adapter from {lora_path}")
        pipe.load_lora_weights(
            LORA_DIR,
            weight_name=LORA_NAME,
            adapter_name=ADAPTER_NAME,
        )
        # Pre-warm by setting a very small weight to validate dimensions.
        pipe.set_adapters([ADAPTER_NAME], adapter_weights=[0.01])
        HAS_LORA = True
        _log("LoRA adapter loaded and validated.")
    except Exception as exc:
        HAS_LORA = False
        _log(f"Failed to load LoRA: {exc}. Running without it.")

    return HAS_LORA


_load_lora_adapter()


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
    """Normalize SDXL input size to multiples of 8 and at least 512."""
    w, h = img.size
    w = max(512, (w // 8) * 8)
    h = max(512, (h // 8) * 8)
    if (w, h) != img.size:
        _log(f"Resizing input image from {img.size} to {(w, h)}")
        img = img.resize((w, h), Image.LANCZOS)
    return img


def _safe_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        return torch.Generator(device=DEVICE).manual_seed(int(seed))
    except Exception:
        _log(f"Invalid seed provided ({seed}); using random seed.")
        return None


def _nan_check_callback(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")
    if latents is not None and not torch.isfinite(latents).all():
        raise ValueError("Non-finite values detected in latents during diffusion")
    return callback_kwargs


def _disable_lora_if_present():
    if hasattr(pipe, "disable_lora"):
        pipe.disable_lora()
    elif hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([], adapter_weights=[])
        except Exception:
            pass


def _run_pipeline(
    prompt: str,
    img: Image.Image,
    strength: float,
    steps: int,
    guidance_scale: float,
    generator: Optional[torch.Generator],
    lora_scale: float,
    use_lora: bool,
):
    if HAS_LORA and use_lora:
        pipe.set_adapters([ADAPTER_NAME], adapter_weights=[lora_scale])
        _log(f"Running with LoRA scale {lora_scale}")
    else:
        _disable_lora_if_present()
        _log("Running without LoRA")

    autocast_dtype = torch.float16 if DEVICE == "cuda" else torch.bfloat16

    with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=autocast_dtype):
        result = pipe(
            prompt=prompt,
            image=img,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=_nan_check_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

    return result.images[0]


# -------------------------------------------------------
# MAIN HANDLER
# -------------------------------------------------------
def handler(event: Dict[str, Any]):
    inp = event.get("input") or {}

    prompt = str(inp.get("prompt", ""))

    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing base64 image"}

    strength = float(inp.get("strength", 0.35))
    # Safe range for SDXL img2img strength
    strength = _clamp(strength, 0.25, 0.6)

    steps = int(inp.get("steps", 18))
    steps = max(10, min(steps, 50))

    guidance_scale = float(inp.get("guidance_scale", 4.0))
    guidance_scale = _clamp(guidance_scale, 1.0, 7.5)

    requested_lora_scale = float(inp.get("lora_scale", 1.0))
    safe_lora_scale = _clamp(requested_lora_scale, 0.0, 1.25)
    if safe_lora_scale != requested_lora_scale:
        _log(
            f"LoRA scale {requested_lora_scale} is high; clamped to safe value {safe_lora_scale}."
        )

    seed = inp.get("seed")

    try:
        img = decode_image(image_b64)
    except Exception as exc:
        return {"error": f"Failed to decode input image: {exc}"}

    img = _fix_size(img)

    generator = _safe_generator(seed)

    # First attempt: with LoRA if available and requested scale > 0
    use_lora = HAS_LORA and safe_lora_scale > 0
    try:
        out_img = _run_pipeline(
            prompt=prompt,
            img=img,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            lora_scale=safe_lora_scale,
            use_lora=use_lora,
        )
    except Exception as exc:
        _log(f"Pipeline failed with LoRA ({exc}); retrying without LoRA.")
        # Retry without LoRA to avoid returning empty images
        try:
            out_img = _run_pipeline(
                prompt=prompt,
                img=img,
                strength=strength,
                steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                lora_scale=0.0,
                use_lora=False,
            )
            use_lora = False
        except Exception as inner_exc:
            return {"error": f"Pipeline failed without LoRA: {inner_exc}"}

    return {
        "refined_image": encode_image(out_img),
        "meta": {
            "strength": strength,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "lora_scale": safe_lora_scale,
            "seed": seed,
            "device": DEVICE,
            "used_lora": HAS_LORA and use_lora,
        },
    }


runpod.serverless.start({"handler": handler})
