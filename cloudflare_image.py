import os
import base64
import requests


CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

# Good fast default for text-to-image on Workers AI
CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"


def generate_image_cloudflare(
    prompt: str,
    output_path: str,
    seed: int = 1,
    model: str = CLOUDFLARE_IMAGE_MODEL,
    steps: int = 4,
    width: int = 1024,
    height: int = 1024
):
    """
    Generate an image using Cloudflare Workers AI and save it to output_path.
    """

    if not CLOUDFLARE_ACCOUNT_ID:
        raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID")
    if not CLOUDFLARE_API_TOKEN:
        raise ValueError("Missing CLOUDFLARE_API_TOKEN")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}"
    )

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    # Parameters can vary a bit by model.
    # FLUX models accept at least prompt + seed; width/height/steps are commonly used.
    payload = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(f"Cloudflare API error: {data}")

    result = data.get("result", {})

    # Workers AI image responses are commonly returned as base64 image data
    image_b64 = result.get("image")
    if not image_b64:
        raise RuntimeError(f"No image returned from Cloudflare: {data}")

    image_bytes = base64.b64decode(image_b64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    return output_path
