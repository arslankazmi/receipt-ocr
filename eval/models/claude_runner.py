"""
ClaudeRunner — uses claude-sonnet-4-6 via Anthropic API

Requires ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image

from eval.models.base import ModelRunner

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_PROMPT_TEMPLATE = (
    "Extract this receipt as JSON conforming to this schema. "
    "Return JSON only, no markdown fences.\nSchema: {schema_json}"
)

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Please return only a valid JSON object matching the schema, no markdown fences."
)


def _encode_image(img: Image.Image) -> tuple[str, str]:
    """Encode a PIL Image to base64 JPEG. Returns (b64_data, media_type)."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _parse_response(text: str) -> dict | None:
    """Try to parse JSON from model response text, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _build_messages(img_b64: str, media_type: str, schema_json: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": _PROMPT_TEMPLATE.format(schema_json=schema_json),
                },
            ],
        }
    ]


class ClaudeRunner(ModelRunner):
    model_id = "claude-sonnet"

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before using ClaudeRunner: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )

        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = "claude-sonnet-4-6"
        self._max_tokens = 2048

        schema_path = _PROJECT_ROOT / "schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        self._schema_json = json.dumps(schema)

    def _call_model(
        self,
        messages: list[dict],
        extra_messages: list[dict] | None = None,
    ) -> str:
        full_messages = messages + (extra_messages or [])
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=full_messages,
        )
        return resp.content[0].text

    def extract(self, image: Image.Image) -> dict:
        img_b64, media_type = _encode_image(image)
        messages = _build_messages(img_b64, media_type, self._schema_json)
        raw = self._call_model(messages)
        pred = _parse_response(raw)

        if pred is None:
            # Retry once on malformed response
            retry_messages = [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _RETRY_PROMPT},
            ]
            raw2 = self._call_model(messages, retry_messages)
            pred = _parse_response(raw2) or {}

        return pred
