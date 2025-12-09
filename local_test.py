import base64
from PIL import Image
import io
import handler  # this imports your handler.py

# Load test input image
with open("test.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# Build simulated RunPod event
event = {
    "input": {
        "prompt": "soft cinematic portrait",
        "strength": 0.6,
        "steps": 4,
        "image": b64
    }
}

print("Running SDXL refine...")

result = handler.handler(event)

print("Done.")
print(result.keys())

# Save output
if "refined_image" in result:
    out = base64.b64decode(result["refined_image"])
    with open("output.png", "wb") as f:
        f.write(out)
    print("Output saved to output.png")
else:
    print(result)
