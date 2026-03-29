import os
import base64
import requests


CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_IMAGE_MODEL = "@cf/bytedance/stable-diffusion-xl-lightning"


def generate_image_cloudflare(
    prompt: str,
    output_path: str,
    seed: int = 1,
    model: str | None = None,
    steps: int = 4,
    width: int = 1024,
    height: int = 1024,
    negative_prompt: str | None = None,
):
    if not CLOUDFLARE_ACCOUNT_ID:
        raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID")
    if not CLOUDFLARE_API_TOKEN:
        raise ValueError("Missing CLOUDFLARE_API_TOKEN")
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required")

    model = model or CLOUDFLARE_IMAGE_MODEL

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}"
    )

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").lower()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Cloudflare may return raw image bytes directly
    if content_type.startswith("image/"):
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    # Or JSON
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"Unexpected response from Cloudflare. "
            f"Content-Type={content_type}, first_200_bytes={response.text[:200]!r}"
        )

    # Cloudflare wrapper format
    if data.get("success") is False:
        raise RuntimeError(f"Cloudflare API error: {data}")

    result = data.get("result")

    # Sometimes result may be base64 image text
    if isinstance(result, dict):
        image_b64 = result.get("image")
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return output_path

    raise RuntimeError(f"No image returned from Cloudflare: {data}")
