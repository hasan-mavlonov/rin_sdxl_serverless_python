# handler.py

import base64
import traceback

import runpod
from lora_pipeline import refine_sdxl


def handler(event):
    """
    Expected input JSON:

    {
      "input": {
        "prompt": "your prompt",
        "image_b64": "<base64 PNG/JPEG>",
        "strength": 0.55,          # optional
        "steps": 30,               # optional
        "guidance_scale": 2.0      # optional
      }
    }
    """
    try:
        inp = event.get("input", {})

        prompt = inp.get("prompt")
        image_b64 = inp.get("image_b64")

        if not prompt or not image_b64:
            return {
                "status": "error",
                "message": "Both 'prompt' and 'image_b64' are required."
            }

        strength = float(inp.get("strength", 0.55))
        steps = int(inp.get("steps", 30))
        guidance_scale = float(inp.get("guidance_scale", 2.0))

        image_bytes = base64.b64decode(image_b64)

        result_png_bytes = refine_sdxl(
            image_bytes=image_bytes,
            prompt=prompt,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

        result_b64 = base64.b64encode(result_png_bytes).decode("utf-8")

        return {
            "status": "success",
            "image_b64": result_b64,
            "meta": {
                "strength": strength,
                "steps": steps,
                "guidance_scale": guidance_scale,
            },
        }

    except Exception as e:
        print("[SDXL-LoRA] ERROR:", e)
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
        }


runpod.serverless.start({"handler": handler})
