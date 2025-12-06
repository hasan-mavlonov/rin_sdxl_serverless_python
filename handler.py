import runpod
import base64
from sdxl_inference.lora_pipeline import refine_image

def handler(event):
    inp = event["input"]

    prompt = inp.get("prompt", "")
    strength = float(inp.get("strength", 0.55))
    steps = int(inp.get("steps", 30))

    image_b64 = inp["image"]
    image_bytes = base64.b64decode(image_b64)

    out_bytes = refine_image(image_bytes, prompt, strength, steps)

    return {
        "status": "success",
        "image": base64.b64encode(out_bytes).decode()
    }

runpod.serverless.start({"handler": handler})
