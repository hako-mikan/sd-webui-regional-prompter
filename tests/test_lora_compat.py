from contextlib import closing
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("mode", ["Attention", "Latent"])
@pytest.mark.parametrize("llm", [False, True])
def test_global_lora_uses_native_loader(host, processing, run_process, mode, llm):
    if llm:
        host.shared.sd_model.__class__ = type("Anima", (), {})
    native = host.networks.load_networks
    linear = host.torch.nn.Linear.forward
    run_process(mode)
    assert host.script.active
    assert not host.script.region_lora
    assert host.networks.load_networks is native
    assert not hasattr(host.networks, "load_lora_state_dict")
    assert host.dynamic_args["online_lora"] is False
    assert host.torch.nn.Linear.forward is linear
    assert processing.all_prompts[0].count("<lora:style:1>") == 1
    assert len(processing.all_prompts[0].split("AND" if mode == "Latent" or llm else "BREAK")) == 5
    if llm:
        assert host.script.ppt == [0] * 5
    else:
        assert len(host.script.ppt) == 5
    host.rp.unloader(host.script, processing)
    assert host.networks.load_networks is native


def test_multiple_global_loras_preserve_weights(host, processing, run_process):
    processing.all_prompts[0] = processing.prompt = processing.prompt.replace(
        "<lora:style:1>", "<lora:style:0.7><lora:second:1.2>")
    run_process("Latent")
    _, data = host.extra.parse_prompts(processing.all_prompts)
    assert [p.items for p in data["lora"]] == [["style", "0.7"], ["second", "1.2"]]
    assert host.networks.load_networks(["style", "second"], [0.7, 1.2], [0.7, 1.2]) == {}
    host.networks.load_networks.assert_called_once_with(["style", "second"], [0.7, 1.2], [0.7, 1.2])
    processing.close()


def regional_setup(host, legacy=False):
    if legacy:
        host.networks.load_lora_state_dict = Mock()
    script = host.script
    script.region_lora = True
    script.debug = False
    script.isbefore15 = False
    script.layer_name = "network_layer_name"
    script.ui_version = 100
    script.slowlora = False
    host.latent.setuploras(script)


def test_neo_regional_loader_delegates_and_preserves_error_contract(host):
    native = host.networks.load_networks
    failures = {"missing": "LoRA does not exist"}
    native.return_value = failures
    regional_setup(host)
    result = host.networks.load_networks(["a", "a", "b", "missing"], [0.6] * 4, [0.8] * 4)
    assert result is failures
    assert dict(result) == failures
    native.assert_called_once_with(["a", "b", "missing"], [1.0] * 3, [1.0] * 3)
    host.latent.unloadlorafowards(host.script)
    assert host.networks.load_networks is native


def test_native_loader_errors_are_not_swallowed(host):
    host.networks.load_networks.side_effect = RuntimeError("broken adapter")
    regional_setup(host)
    with pytest.raises(RuntimeError, match="broken adapter"):
        host.networks.load_networks(["a"], [1], [1])
    host.latent.unloadlorafowards(host.script)


def test_legacy_forge_keeps_regional_loader(host):
    native = host.networks.load_networks
    regional_setup(host, legacy=True)
    host.networks.load_networks(["a"], [0.6], [0.8])
    native.assert_called_once_with(["a"], [1.0], [1.0])
    host.networks.load_lora_state_dict.assert_not_called()
    host.latent.unloadlorafowards(host.script)
    assert host.networks.load_networks is native


@pytest.mark.parametrize("host", ["v1.10.1", "classic 1.10.1"], indirect=True)
def test_a1111_and_reforge_keep_their_native_loader(host, processing, run_process):
    native = host.networks.load_networks
    linear = host.torch.nn.Linear.forward
    run_process("Latent")
    assert host.script.active
    assert host.networks.load_networks is native
    assert host.torch.nn.Linear.forward is not linear
    host.rp.unloader(host.script, processing)
    assert host.torch.nn.Linear.forward is linear
    assert host.networks.load_networks is native


def test_regional_weights_are_selected_and_restored(host):
    regional_setup(host)
    a, b = [1.0, "a patch"], [1.0, "b patch"]
    module = NS(forge_online_loras={"weight": [a, b]})
    host.shared.sd_model.forge_objects.unet.model.named_modules = lambda: [("layer", module)]
    regioner = host.latent.regioner
    regioner.u_llist = [{"a": 0.6, "b": 0}, {"a": 0, "b": 0.8}]
    regioner.set_region(0)
    assert [a[0], b[0]] == [0.6, 0]
    regioner.set_region(1)
    assert [a[0], b[0]] == [0, 0.8]
    host.latent.unloadlorafowards(host.script)
    assert [a[0], b[0]] == [1.0, 1.0]


def test_native_regional_prompt_weights_keep_te_and_unet_separate(host):
    regional_setup(host)
    host.script.current_prompts = ["girl <lora:a:0.6:0.8> AND girl <lora:b:1:te=0.3:unet=0.9>"]
    host.script.lstop = host.script.lstop_hr = 100
    for name in ("a", "b"):
        host.networks.loaded_networks.append(NS(name=name, te_multiplier=1, unet_multiplier=1,
                                               network_on_disk=NS(filename=name, alias=name, name=name)))
    host.latent.lora_namer(host.script, None, "0", "0")
    assert host.latent.regioner.te_llist == [{"a": 0, "b": 0}, {"a": 0.6, "b": 0}, {"a": 0, "b": 0.3}]
    assert host.latent.regioner.u_llist == [{"a": 0.8, "b": 0}, {"a": 0, "b": 0.9}, {"a": 0, "b": 0}]
    host.latent.unloadlorafowards(host.script)


@pytest.mark.parametrize("online", [False, True])
def test_repeated_install_and_unload_restores_owned_state(host, online):
    host.dynamic_args["online_lora"] = online
    native = host.networks.load_networks
    linear = host.torch.nn.Linear.forward
    te_forward = host.te.forward
    for _ in range(3):
        regional_setup(host)
        installed = host.networks.load_networks
        regional_setup(host)
        assert host.networks.load_networks is installed
        assert len(host.latent.saved_te_forwards) == 1
        host.latent.unloadlorafowards(host.script)
        host.latent.unloadlorafowards(host.script)
        assert host.networks.load_networks is native
        assert host.torch.nn.Linear.forward is linear
        assert host.te.forward == te_forward
        assert not hasattr(host.te, "rp_original_forward")
        assert host.dynamic_args["online_lora"] is online
        assert host.shared.opts.lora_functional is False
        assert "model_function_wrapper" not in host.shared.sd_model.forge_objects.unet.model_options


def test_exception_closes_generation_and_restores_hooks(host, processing):
    from scripts.lora_compat import generation_cleanup
    native = host.networks.load_networks
    close = processing.close
    generation_cleanup(host.script, processing, host.rp.unloader)
    with pytest.raises(RuntimeError, match="sampling failed"):
        with closing(processing):
            regional_setup(host)
            raise RuntimeError("sampling failed")
    close.assert_called_once()
    assert processing.close is close
    assert host.networks.load_networks is native
    assert host.dynamic_args["online_lora"] is False
    assert not host.latent.lactive


def test_disabled_plugin_leaves_native_loras_and_options_alone(host, processing, run_process):
    native = host.networks.load_networks
    host.networks.loaded_networks.append("already loaded")
    host.shared.opts.lora_functional = True
    host.dynamic_args["online_lora"] = True
    run_process(active=False)
    assert host.networks.loaded_networks == ["already loaded"]
    assert host.shared.opts.lora_functional is True
    assert host.dynamic_args["online_lora"] is True
    assert host.networks.load_networks is native


def test_unsupported_llm_regional_lora_is_explicit_and_cleans_up(host, processing, run_process):
    host.shared.sd_model.__class__ = type("Anima", (), {})
    processing.all_prompts[0] = processing.prompt = "1girl <lora:a:1> ADDCOL 1girl <lora:b:1>"
    with pytest.raises(ValueError, match="regional LoRAs are not supported"):
        run_process("Latent")
    assert not host.script.active
    assert host.script._rp_cleanup is None
    assert not host.latent.region_lora_active
    assert processing.extra_generation_params["RP Active"] is False
    assert "not supported" in processing.extra_generation_params["RP LoRA Error"]


def test_latent_to_attention_then_disabled_restores_native_loader(host, processing, run_process):
    native = host.networks.load_networks
    regional_setup(host)
    run_process("Attention")
    assert host.networks.load_networks is native
    host.rp.unloader(host.script, processing)
    run_process(active=False)
    assert host.networks.load_networks is native


def test_sampler_hook_is_unique_and_modifiers_are_removed(host):
    native = host.sf.calc_cond_uncond_batch
    host.latent.hook_neo_cond_batch(host.script)
    installed = host.sf.calc_cond_uncond_batch
    host.latent.hook_neo_cond_batch(host.script)
    assert host.sf.calc_cond_uncond_batch is installed
    assert host.shared.sd_model.forge_objects.unet.model_options["conditioning_modifiers"] == [host.latent.normalise_cond_strengths]
    host.latent.unhook_neo_cond_batch()
    assert host.sf.calc_cond_uncond_batch is native
    assert host.shared.sd_model.forge_objects.unet.model_options["conditioning_modifiers"] == []


def test_region_preview_does_not_overflow_uint8(host):
    import warnings
    from scripts.regions import makeimgtmp
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        image, _ = makeimgtmp("1,1,1,1,1", "Columns", True, False, False, 384, 640)
    assert image.size == (640, 384)


def test_attention_hooks_preserve_instance_forward_across_repeated_generations(host, processing):
    class CrossAttention:
        def forward(self, x):
            return x

    layer = CrossAttention()
    original = lambda x: x
    layer.forward = original
    host.shared.sd_model.forge_objects.unet.model.named_modules = lambda: [("attn2", layer)]
    for _ in range(3):
        host.attention.hook_forwards(host.script, processing)
        installed = layer.forward
        host.attention.hook_forwards(host.script, processing)
        assert layer.forward is installed
        host.attention.hook_forwards(host.script, processing, remove=True)
        assert layer.forward is original


def test_original_model_and_cloned_wrappers_restored_after_model_switch(host):
    old_model = host.shared.sd_model
    original = lambda apply, params: apply
    old_model.forge_objects.unet.model_options["model_function_wrapper"] = original
    regional_setup(host)
    clone = host.Patcher()
    clone.model_options = dict(old_model.forge_objects.unet.model_options)
    old_model.forge_objects_after_applying_lora = NS(unet=clone)
    host.shared.sd_model = NS(forge_objects=NS(unet=host.Patcher()))
    host.latent.unloadlorafowards(host.script)
    assert old_model.forge_objects.unet.model_options["model_function_wrapper"] is original
    assert clone.model_options["model_function_wrapper"] is original
    assert host.shared.sd_model.forge_objects.unet.model_options == {}


def test_script_reload_cleans_generation(host, processing, run_process):
    close = processing.close
    run_process("Latent")
    cleanup = host.callbacks.on_script_unloaded.call_args.args[0]
    cleanup()
    assert processing.close is close
    assert not host.script.active
    assert not host.latent.lactive
    host.callbacks.remove_callbacks_for_function.assert_called_once_with(cleanup)


@pytest.mark.parametrize("result", [None, {}, {"missing": "not found"}])
def test_adapter_preserves_backend_return_and_optional_arguments(host, result):
    from scripts.lora_compat import regional_loader
    original = Mock(return_value=result)
    adapted = regional_loader(original)
    assert adapted(["a"], [0.6], [0.8], [16], future_option=True) is result
    original.assert_called_once_with(["a"], [1.0], [1.0], [16], future_option=True)


def test_absent_lora_functional_option_is_not_added(host):
    del host.shared.opts.lora_functional
    regional_setup(host)
    host.latent.unloadlorafowards(host.script)
    assert not hasattr(host.shared.opts, "lora_functional")


def test_batch_option_restored_to_generation_value(host, processing, run_process):
    host.shared.opts.batch_cond_uncond = False
    run_process("Latent")
    processing.close()
    assert host.shared.opts.batch_cond_uncond is False


def test_new_neo_regional_patch_format_is_not_silently_treated_as_supported(host, processing, run_process):
    host.shared.sd_model.forge_objects.clip.patcher = NS(weight_wrapper_patches={})
    processing.all_prompts[0] = processing.prompt = "1girl <lora:a:1> ADDCOL 1girl <lora:b:1>"
    native = host.networks.load_networks
    with pytest.raises(ValueError, match="patch format does not support"):
        run_process("Latent")
    assert host.networks.load_networks is native
    assert not host.script.active
    assert host.dynamic_args["online_lora"] is False


def test_cleanup_failure_still_restores_lora_loader(host, processing, monkeypatch):
    from scripts.lora_compat import generation_cleanup
    native = host.networks.load_networks
    generation_cleanup(host.script, processing, host.rp.unloader)
    regional_setup(host)
    host.script.hooked = True
    monkeypatch.setattr(host.rp, "hook_forwards", Mock(side_effect=RuntimeError("attention cleanup failed")))
    with pytest.raises(RuntimeError, match="attention cleanup failed"):
        processing.close()
    assert host.networks.load_networks is native
    assert host.dynamic_args["online_lora"] is False
    assert not host.script.active


def test_module_attention_hooks_restore_each_original(host, monkeypatch):
    import sys
    import types
    backend_attention = types.ModuleType("backend.attention")
    backend_attention.attention_function = Mock()
    backend_attention.attention_pytorch = Mock()
    monkeypatch.setitem(sys.modules, "backend.attention", backend_attention)
    monkeypatch.setattr(sys.modules["backend"], "attention", backend_attention, raising=False)
    modules = []
    for name in ("backend.nn.krea", "backend.nn.lumina"):
        module = types.ModuleType(name)
        module.attention_function = Mock()
        monkeypatch.setitem(sys.modules, name, module)
        modules.append((module, module.attention_function))
    host.attention.hook_masked_attention()
    host.attention.hook_masked_attention()
    assert len(host.attention.saved_attention_functions) == 2
    for module, original in modules:
        module.attention_function(1, 2, 3, 4)
        original.assert_called_once_with(1, 2, 3, 4, mask=None)
    host.attention.hook_masked_attention(remove=True)
    for module, original in modules:
        assert module.attention_function is original
