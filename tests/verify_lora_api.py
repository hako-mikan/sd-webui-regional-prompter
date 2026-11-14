"""Optional GPU regression against a running WebUI API (not part of CI).

python tests/verify_lora_api.py --url http://127.0.0.1:7862 --lora NAME --output DIRECTORY
Use an isolated WebUI settings file and inspect its console for loading errors.
Pixel differences establish an inference effect, not style quality or isolation.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--lora", required=True)
    parser.add_argument("--second-lora")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    regions = "1girl, pink hair ADDCOL 1girl, red hair ADDCOL 1girl, blonde hair ADDCOL 1girl, long blonde hair ADDCOL 1girl, blue hair"
    common = ("multiple girls, five adult women, fully clothed, modest dresses, standing in a row, "
              "masterpiece, best quality, @4x0style, ")
    lora = f"<lora:{args.lora}:1>"
    images = {}
    summary = {}

    def generate(name, mode, tags):
        regional = mode is not None
        prompt = common + tags + (" ADDCOMM " + regions if regional else ", 1girl, pink hair")
        rp_args = [regional, False, "Matrix", "Columns", "Mask", "Prompt", "1,1,1,1,1", "0",
                   False, False, False, mode or "Attention", [], "0", "0", "0.4", None, "0", "0", False]
        payload = dict(prompt=prompt, negative_prompt="worst quality, low quality, nude, nudity, nsfw", seed=123456789,
                       sampler_name="Euler", scheduler="Simple", steps=8, cfg_scale=4, width=640,
                       height=384, batch_size=1, n_iter=1, save_images=False,
                       alwayson_scripts={"regional prompter": {"args": rp_args}})
        (args.output / f"{name}_request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        request = Request(args.url.rstrip("/") + "/sdapi/v1/txt2img", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=900) as response:
            result = json.load(response)
        raw = base64.b64decode(result.pop("images")[0])
        (args.output / f"{name}.png").write_bytes(raw)
        (args.output / f"{name}_response.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        images[name] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int16)
        info = json.loads(result["info"])
        extra = info["extra_generation_params"]
        assert bool(extra.get("RP Active")) == regional, (name, extra)
        if tags:
            assert args.lora in extra.get("Lora hashes", ""), (name, extra)
        summary[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "extra_generation_params": extra}
        print(name, "finished", flush=True)

    generate("native_off", None, "")
    generate("native_on_before", None, lora)
    for mode in ("Attention", "Latent"):
        prefix = mode.lower()
        generate(prefix + "_off", mode, "")
        generate(prefix + "_on", mode, lora)
        generate(prefix + "_repeat", mode, lora)
        if args.second_lora:
            generate(prefix + "_two_loras", mode, lora + f"<lora:{args.second_lora}:0.5>")
    generate("native_on_after", None, lora)
    for prefix, off, on in [("native", "native_off", "native_on_before"),
                            ("attention", "attention_off", "attention_on"),
                            ("latent", "latent_off", "latent_on")]:
        difference = np.abs(images[off] - images[on])
        assert difference.any(), f"{prefix}: LoRA has no detectable effect"
        summary[prefix + "_effect"] = {"mean_absolute_pixel_difference": float(difference.mean()),
                                       "changed_pixel_fraction": float(np.any(difference, axis=-1).mean())}
    for first, second in [("attention_on", "attention_repeat"), ("latent_on", "latent_repeat"),
                          ("native_on_before", "native_on_after")]:
        assert np.array_equal(images[first], images[second]), f"Images differ: {first}, {second}"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("All inference comparisons passed", flush=True)


if __name__ == "__main__":
    main()
