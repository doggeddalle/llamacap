from __future__ import annotations

import base64
import json
import logging
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path

from llamacap.errors import CaptionGenerationError
from llamacap.profiles import Profile
from llamacap.server import LlamaServer

logger = logging.getLogger("llamacap")


def build_payload(profile: Profile, image_path: Path) -> dict:
    gen = profile.generation
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "application/octet-stream"
    b64_data = base64.b64encode(image_path.read_bytes()).decode("ascii")

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": profile.prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                    },
                ],
            }
        ],
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "top_k": gen.top_k,
        "repeat_penalty": gen.repeat_penalty,
        "seed": gen.seed,
        "max_tokens": gen.n_predict,
        "stream": False,
    }


def generate_caption(
    server: LlamaServer,
    profile: Profile,
    image_path: Path,
    timeout_seconds: int,
) -> str:
    payload = build_payload(profile, image_path)
    logger.debug("request payload (image omitted): %s", {**payload, "messages": "<omitted>"})

    request = urllib.request.Request(
        f"{server.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        excerpt = e.read().decode("utf-8", errors="replace").strip()[-500:]
        raise CaptionGenerationError(
            f"llama-server returned HTTP {e.code}: {excerpt}"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CaptionGenerationError(f"request to llama-server failed: {e}") from e
    except json.JSONDecodeError as e:
        raise CaptionGenerationError(f"llama-server returned invalid JSON: {e}") from e

    try:
        caption = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as e:
        raise CaptionGenerationError(f"unexpected llama-server response shape: {e}") from e

    if not caption:
        raise CaptionGenerationError("model produced no output")

    return caption
