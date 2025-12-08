import base64
import io
from typing import Any, Dict

import runpod
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Determine device and dtype once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the pipeline globally to avoid reloading per request
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=DTYPE,
).to(DEVICE)


def decode_image(image_b64: str) -> Image.Image:
    """Decode a base64 string to a PIL image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(image: Image.Image) -> str:
    """Encode a PIL image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_data = event.get("input", {}) or {}

    prompt = input_data.get("prompt", "")
    strength = float(input_data.get("strength", 0.55))
    steps = int(input_data.get("steps", 20))
    image_b64 = input_data.get("image")

    if not image_b64:
        return {"error": "Missing required field: image"}

    try:
        init_image = decode_image(image_b64)
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": f"Failed to decode image: {exc}"}

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
        ).images[0]

    refined_b64 = encode_image(result)

    return {"refined_image": refined_b64}


runpod.serverless.start({"handler": handler})
