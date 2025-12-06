import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to(device)

pipe.load_lora_weights("/app/lora/rinxl_lora.safetensors")

def refine_image(image_bytes, prompt, strength=0.55, steps=30):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    result = pipe(
        prompt=prompt,
        image=img,
        strength=strength,
        num_inference_steps=steps
    ).images[0]

    buf = BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
