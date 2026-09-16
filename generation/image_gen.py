"""
Image generation — switches between two backends:

  USE_OPENAI=false (default): Stable Diffusion XL via diffusers (local, free, needs GPU)
  USE_OPENAI=true            : DALL-E 3 via OpenAI API (~$0.04 per image)

Output is always saved to outputs/generated_outfit.png and the path returned.
"""

import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = str(OUTPUT_DIR / "generated_outfit.png")


def generate_outfit_image(prompt: str) -> str:
    """
    Generate an outfit image from a text prompt.
    Returns the file path of the saved image.
    """
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    if use_openai:
        return _generate_dalle3(prompt)
    else:
        return _generate_sdxl(prompt)


# ── Backend A: DALL-E 3 (OpenAI) ──────────────────────────────────────────────

def _generate_dalle3(prompt: str) -> str:
    import requests
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    # Download and save locally
    img_data = requests.get(image_url).content
    with open(OUTPUT_PATH, "wb") as f:
        f.write(img_data)

    print(f"DALL-E 3 image saved to {OUTPUT_PATH}")
    return OUTPUT_PATH


# ── Backend B: Stable Diffusion XL (local) ────────────────────────────────────

def _generate_sdxl(prompt: str) -> str:
    import torch
    from diffusers import StableDiffusionXLPipeline

    print("Loading SDXL pipeline (first run will download ~7GB)...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # reduces VRAM usage

    # Fashion-specific negative prompt to avoid common artefacts
    negative_prompt = (
        "blurry, low quality, distorted face, extra limbs, "
        "bad anatomy, watermark, text, logo, cartoon"
    )

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    image.save(OUTPUT_PATH)
    print(f"SDXL image saved to {OUTPUT_PATH}")

    # Free VRAM after generation
    del pipe
    torch.cuda.empty_cache()

    return OUTPUT_PATH
