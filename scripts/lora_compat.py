"""Small adapters around the host's LoRA loader; never read model files here."""
import re
from functools import wraps


LORA_TAG = re.compile(r"<lora:([^>]+)>")


def regional_loras(prompts, separator="AND"):
    """Identical LoRA tags in every region describe a global LoRA."""
    for prompt in prompts:
        regions = [tuple(sorted(LORA_TAG.findall(part))) for part in prompt.split(separator)]
        if any(tags != regions[0] for tags in regions[1:]):
            return True
    return False


def deduplicate_loras(prompt):
    # ADDCOMM repeats the tags, but a global adapter must only be loaded once.
    seen = set()

    def unique(match):
        tag = match.group(1)
        if tag in seen:
            return ""
        seen.add(tag)
        return match.group(0)

    return LORA_TAG.sub(unique, prompt)


def regional_loader(original):
    """Load each adapter at unit strength, then let LoRARegioner select weights.

    Keep Neo's file handling, model mappings, quantization and error return value.
    The region weights must not multiply an already weighted patch a second time.
    """
    @wraps(original)
    def load_networks(names, te_multipliers=None, unet_multipliers=None, *args, **kwargs):
        names = list(dict.fromkeys(names))
        return original(names, [1.0] * len(names), [1.0] * len(names), *args, **kwargs)

    return load_networks


def supports_region_patches(model):
    """LoRARegioner uses Forge's lora_patches/lora_loader layout.

    Recent Neo uses weight_wrapper_patches instead. Loading unit patches there
    without a matching region switcher would silently apply every region LoRA.
    """
    clip = model.forge_objects.clip
    return hasattr(clip.patcher, "lora_patches") and hasattr(clip.patcher, "lora_loader")


def generation_cleanup(script, processing, unload):
    """The UI and API close processing even when postprocess is skipped on error."""
    from modules import script_callbacks
    original_close = processing.close

    def cleanup():
        if script._rp_cleanup is not cleanup:
            return
        script._rp_cleanup = None
        processing.close = original_close
        script_callbacks.remove_callbacks_for_function(cleanup)
        unload(script, processing)

    @wraps(original_close)
    def close():
        try:
            cleanup()
        finally:
            original_close()

    script._rp_cleanup = cleanup
    processing.close = close
    script_callbacks.on_script_unloaded(cleanup)
    return cleanup
