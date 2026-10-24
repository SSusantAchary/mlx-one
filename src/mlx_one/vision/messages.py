"""Preserve ordered image content while applying the checkpoint chat template."""

from __future__ import annotations

import base64
import binascii


def prepare_messages(bundle, messages):
    normalized, images = [], []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError("unsupported vision message role")
        if isinstance(content, str):
            text = content
        elif isinstance(content, (list, tuple)):
            parts = []
            for part in content:
                kind = part.get("type")
                if kind == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif kind in {"image", "image_url"}:
                    if role != "user":
                        raise ValueError("images are supported in user messages only")
                    image = (
                        part.get("image")
                        if kind == "image"
                        else part.get("image_url", {}).get("url")
                    )
                    if isinstance(image, str) and image.startswith("data:"):
                        header, separator, payload = image.partition(",")
                        if not separator or header not in {
                            "data:image/png;base64",
                            "data:image/jpeg;base64",
                            "data:image/webp;base64",
                        }:
                            raise ValueError("unsupported image data URL")
                        if len(payload) > 28 * 1024**2:
                            raise ValueError("image data URL exceeds the size budget")
                        try:
                            image = base64.b64decode(payload, validate=True)
                        except binascii.Error as exc:
                            raise ValueError("invalid image base64") from exc
                    elif kind == "image_url":
                        raise ValueError("remote image URLs are disabled; use a base64 data URL")
                    images.append(image)
                    parts.append("<image>")
                else:
                    raise ValueError("only text and static image content is supported")
            text = "".join(parts)
        else:
            raise ValueError("vision message content must be text or an array")
        normalized.append(
            {
                "role": role,
                "content": text,
                "thinking": message.get("thinking", ""),
                "reasoning": message.get("reasoning", ""),
                "reasoning_content": message.get("reasoning_content", ""),
            }
        )
    if len(images) > 8:
        raise ValueError("at most eight images are supported")
    if bundle.chat_template is None:
        raise ValueError("checkpoint chat template is required for image chat")
    return bundle.chat_template.render(
        normalized, enable_thinking=False, add_vision_id=False
    ), tuple(images)
