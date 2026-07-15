from __future__ import annotations

from llamacap.errors import ProfileError
from llamacap.profiles import TriggerWordConfig


def apply_trigger_word(caption: str, trigger: TriggerWordConfig) -> str:
    if not trigger.value:
        return caption

    if trigger.placement == "prefix_comma":
        return f"{trigger.value}, {caption}"
    if trigger.placement == "prefix_period":
        return f"{trigger.value}. {caption}"
    if trigger.placement == "none":
        return caption

    raise ProfileError(f"Unknown trigger_word.placement: {trigger.placement!r}")
