"""Import the real extension with small host doubles, without torch or models."""
import importlib
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import PIL.ImageDraw


@pytest.fixture
def host(monkeypatch, tmp_path, request):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    for name in list(sys.modules):
        if name == "scripts" or name.startswith("scripts."):
            monkeypatch.delitem(sys.modules, name)

    class Linear:
        def forward(self, x):
            return x

    torch = module("torch", nn=NS(Linear=Linear, Module=object), device=lambda x: x,
                   float32="fp32", float16="fp16", bfloat16="bf16", einsum=Mock())
    tv = module("torchvision")
    transforms = module("torchvision.transforms", InterpolationMode=Mock(), Resize=Mock())
    tv.transforms = transforms
    transforms.functional = module("torchvision.transforms.functional")
    module("einops", rearrange=Mock(), repeat=Mock())
    module("cv2")
    module("gradio", Checkbox=type("Checkbox", (), {}), __version__="4.0.0", update=lambda **kw: kw)

    class Patcher:
        def __init__(self):
            self.model_options = {}
            self.model = NS(named_modules=lambda: [])

        def set_model_unet_function_wrapper(self, wrapper):
            self.model_options["model_function_wrapper"] = wrapper

        def add_conditioning_modifier(self, modifier, ensure_uniqueness=False):
            values = self.model_options.setdefault("conditioning_modifiers", [])
            if modifier not in values:
                values.append(modifier)

    class Tripwire:
        def forward(self, x):
            return x

    te = Tripwire()
    encoder = NS(named_modules=lambda: [("layers.0.self_attn.q_proj", te)])
    clip = NS(cond_stage_model=NS(named_children=lambda: [("clip_l", encoder)]),
              patcher=NS(lora_patches={}, lora_loader=NS()))
    objects = NS(unet=Patcher(), clip=clip)
    model = type("StableDiffusionXL", (), {})()
    model.forge_objects = model.forge_objects_original = model.forge_objects_after_applying_lora = objects
    tokenizer = NS(tokenize_line=lambda text: ([NS(tokens=[1, 49407])], len(text.split())))
    model.text_processing_engine_l = tokenizer
    model.conditioner = NS(embedders=[tokenizer])
    model.cond_stage_model = tokenizer
    model.model = NS(diffusion_model=objects.unet.model)
    shared = module("modules.shared", sd_model=model,
                    opts=NS(lora_functional=False, batch_cond_uncond=True, data={}),
                    cmd_opts=NS(no_prompt_history=True))
    params_type = type("Params", (), {})
    callbacks = module("modules.script_callbacks", CFGDenoisedParams=params_type,
                       CFGDenoiserParams=params_type, AfterCFGCallbackParams=params_type,
                       on_cfg_after_cfg=Mock(), on_cfg_denoised=Mock(), on_cfg_denoiser=Mock(),
                       on_ui_settings=Mock(), on_script_unloaded=Mock(), remove_callbacks_for_function=Mock())
    launch_utils = module("modules.launch_utils", git_tag=lambda: getattr(request, "param", "neo 2.31"))
    hijack = module("modules.sd_hijack", model_hijack=NS(embedding_db=NS(register_embedding_by_name=Mock())))
    module("launch", git_tag=launch_utils.git_tag)
    paths = module("modules.paths", data_path=str(tmp_path), script_path=str(tmp_path))
    scripts = module("modules.scripts", Script=type("Script", (), {}), basedir=lambda: str(tmp_path))
    module("modules.ui")
    module("modules.processing", Processed=Mock())

    class ExtraNetworkParams:
        def __init__(self, items):
            self.items = items
            self.positional = [x for x in items if "=" not in x]
            self.named = dict(x.split("=", 1) for x in items if "=" in x)

    def parse_prompt(prompt):
        from collections import defaultdict
        parsed = defaultdict(list)

        def found(match):
            parsed[match[1]].append(ExtraNetworkParams(match[2].split(":")))
            return ""

        return re.sub(r"<(\w+):([^>]+)>", found, prompt), parsed

    def parse_prompts(prompts):
        values = [parse_prompt(p) for p in prompts]
        return [p[0] for p in values], values[0][1]

    extra = module("modules.extra_networks", parse_prompt=parse_prompt, parse_prompts=parse_prompts)
    parser = module("modules.prompt_parser", get_learned_conditioning_prompt_schedules=lambda p, steps: [[[steps, p[0]]]])
    module("modules", shared=shared, devices=NS(), launch_utils=launch_utils,
           extra_networks=extra, paths=paths, scripts=scripts, prompt_parser=parser, sd_hijack=hijack,
           script_callbacks=callbacks)

    dynamic_args = {"online_lora": False}
    module("backend")
    module("backend.args", dynamic_args=dynamic_args)
    sf = module("backend.sampling.sampling_function", calc_cond_uncond_batch=lambda *a: None)
    module("backend.sampling", sampling_function=sf)
    networks = module("networks", load_networks=Mock(return_value={}), loaded_networks=[])
    module("lora", loaded_loras=networks.loaded_networks)
    latent = importlib.import_module("scripts.latent")
    attention = importlib.import_module("scripts.attention")
    rp = importlib.import_module("scripts.rp")
    for name in ("savepresets", "neighbor", "thresholddealer", "bratioprompt", "hrdealer"):
        monkeypatch.setattr(rp, name, Mock())
    script = rp.Script()
    return NS(latent=latent, attention=attention, rp=rp, script=script, networks=networks,
              shared=shared, torch=torch, dynamic_args=dynamic_args, sf=sf, te=te,
              extra=extra, callbacks=callbacks, Patcher=Patcher)


@pytest.fixture
def processing(host):
    prompt = ("multiple girls, masterpiece, best quality, @4x0style, <lora:style:1> ADDCOMM "
              "1girl, pink hair ADDCOL 1girl, red hair ADDCOL 1girl, blonde hair ADDCOL "
              "1girl, long blonde hair ADDCOL 1girl, blue hair")
    return NS(prompt=prompt, negative_prompt="", all_prompts=[prompt], all_negative_prompts=[""],
              extra_generation_params={}, sd_model=host.shared.sd_model, height=384, width=640,
              batch_size=1, sampler_name="Euler", steps=8, close=Mock())


@pytest.fixture
def run_process(host, processing):
    def run(mode="Attention", active=True):
        return host.script.process(processing, active, False, "Matrix", "Columns", "Mask", "Prompt",
                                   "1,1,1,1,1", "0", False, False, False, mode, [], "0", "0",
                                   "0.4", None, "0", "0", False)
    return run
