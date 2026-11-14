# Forge LoRA compatibility and regression testing

On Forge, global LoRAs in Attention and Latent mode use the host's loader and weights.
Identical LoRA tags repeated by `ADDCOMM` are removed after common text expansion,
so five regions do not load or accumulate five copies of the adapter. Region
text, separators, attention masks and conditioning normalization are unchanged.

## Root cause

The former `scripts/latent.py:load_networks` was a copy of an older Forge loader
(introduced in c572732e). `setuploras` replaced `networks.load_networks` whenever
Latent mode was selected, including global-only LoRAs and LLM-based models that
do not support regional LoRAs. Unloading restored `Linear.forward` and, recently,
`online_lora`, but never restored the loader or the text-encoder tripwire. A
later Attention or disabled run could therefore inherit the Latent loader. A
fresh Attention-only session does not install that loader.

This is stale API duplication plus a hook lifetime problem, not incorrect Neo
version detection. On the inspected Neo implementation:

- `networks.load_lora_state_dict` no longer exists. The native loader uses
  `load_torch_file`, performs model-specific mappings (including Anima), and
  creates the appropriate native patchers.
- `ExtraNetworkLora.activate` calls `self.errors.update(load_networks(...))`.
  The copied loader returns `None` even on successful, cached, empty and no-model
  paths. This is a separate return-contract bug.
- `modules.extra_networks.activate` catches the first activation failure and
  retries the still-unactivated extra network with `[]`. The copied loader then
  returns `None`, producing the reported `TypeError`. Merely replacing the missing
  state-dict reader would leave this bug in place.
- The copied loader also writes the cache hash before reading the file and forces
  online mode off based on storage dtype. RP should not maintain another version
  of the host's model loading and quantization rules.

`ADDCOMM` independently replicates the LoRA tag into every region. Neo's offline
patcher appends patches, so repeated tags can accumulate strength as well as
cause redundant file reads. The global path now passes each identical tag once.
Different adapters and different tag arguments are not combined.

`Pos tokens: [0, ...]` is an intentional placeholder in the LLM-encoder branch
of `tokendealer`. Those models use separate conditionings rather than CLIP chunk
spans. It does not indicate empty text or a LoRA activation failure.

## Regional LoRAs and backend boundaries

On Forge patchers supported by the existing `LoRARegioner` (`lora_patches` and
`lora_loader`), a temporary adapter calls the **native** loader once per adapter
at unit strength. `LoRARegioner` then applies the requested regional TE and U-Net
weights separately. Unit loading prevents multiplying an already scaled patch
by the requested weight a second time. The native return value, errors and
exceptions pass through unchanged. A1111 and reForge keep their native loader
and existing regional forward path.

Anima, Z-Image, Krea2 and the other LLM-encoder models still do not support
regional LoRAs. Recent Neo also replaced `lora_patches/lora_loader` with
`weight_wrapper_patches`/`OnlineLoRAPatch`; the existing region switcher cannot
modify that format. Such requests now emit an explicit error and clean up RP,
rather than loading unit-strength adapters with a non-working region switcher.
Global `ADDCOMM` LoRAs work independently of these regional limitations.

Attention mode does not provide independent regional LoRA weights. Use Latent
mode on a supported backend for that feature. Stop-step controls also require
the supported regional patch format; they are not newly implemented for DiTs.

## Hook lifetime

Normal postprocess, processing-object close (including sampling exceptions),
script unload, and the next process call share an idempotent cleanup. Both the
standard UI and API close their processing object with `contextlib.closing`.
Permanent CFG callback registrations remain in place and are gated by activity,
as before; they are not repeatedly registered per generation.

Cleanup restores the loader, linear forward, exact text-encoder forward,
temporary U-Net wrappers (including clones), sampling batch hook, conditioning
modifier, LoRA strengths, functional option and online mode that RP owns.
Original model instances are retained for cleanup across a model switch.
Inactive/global-only runs do not clear the host's loaded LoRA list or overwrite
its functional/online settings. Existing GGUF/low-bit and regional strength
restoration fixes from #400/#409 are retained.

## Automated tests

```
python -m pip install -r tests/requirements.txt
python -m pytest -q
python -m compileall -q scripts tests
```

The tests import the real extension with lightweight host doubles. No PyTorch,
GPU, diffusion checkpoint or network access is required. They cover global
Attention/Latent paths, multiple/shared adapters, missing old APIs, native return
contracts and exceptions, supported regional switching, separate TE/U-Net
weights, unsupported patch layouts, disabled runs, repeated hook installation,
exception/reload cleanup, model switches, online/functional/batch settings and
NumPy uint8 preview colors. CI runs on Python 3.10, 3.11 and 3.13.

## Optional real inference

Start WebUI with `--nowebui --port 7862` using an isolated copy of its settings.
Choose a compatible checkpoint, text encoder, VAE and LoRA. Then run:

```
python tests/verify_lora_api.py --url http://127.0.0.1:7862 --lora YOUR_LORA --second-lora ANOTHER_LORA --output OUTPUT_DIRECTORY
```

The optional script saves all requests, metadata, PNGs and a summary. It uses
seed 123456789, Euler, Simple scheduler, 8 steps, CFG 4, 640x384 and batch size 1
as an interface smoke test. It checks on/off image differences, exact repeated
outputs in Attention/Latent, and unchanged native LoRA output before/after RP.
Inspect the server logs too: the script cannot detect every error WebUI catches.

Local validation used Windows, Neo 398279fe, PyTorch 2.10.0+cu130 and an RTX 4080
SUPER. Anima-2.9B-preview-v1 int8 convrot, Qwen3 0.6B and Qwen Image VAE were
tested with one and two style adapters, with native on-the-fly mode both off and
on. All smoke-test comparisons passed. The host reported weight 1.0 and 400
matched U-Net keys for the primary adapter, and performed its native 28-to-40
block Anima remapping. This does **not** establish that the remapped adapter is
artistically suitable for the selected checkpoint.

These image comparisons establish an inference effect and repeatability, not
style quality, an exact five-person composition or complete regional isolation.
Low-step smoke images must not be presented as visual acceptance tests. Actual
A1111, Forge, reForge, SD1.5, SDXL, Z-Image, GGUF, fp8 and other DiT checkpoint
runs remain unverified; the host-double tests are not a substitute for them.

A separate 24-step, 1280x768 clothed group-portrait comparison did not achieve
the requested five people: the regional run produced seven, both with and
without LoRA. The unmodified upstream 9d6c1f4 reproduced the same no-LoRA image
**pixel for pixel**, so this is an existing composition limitation in this test,
not a demonstrated regression from the loader fix. Five-region masks and prompt
parsing are tested, but exact character count is not claimed as validated.

The isolated unmodified upstream was also run in Latent mode followed by
Attention with LoRA. Its console reproduced both the missing
`load_lora_state_dict` exception and the subsequent `NoneType` activation error.
WebUI returned HTTP 200 despite these caught errors; checking only HTTP status
would have missed the failure.

Related upstream reports: #400 (regional strengths), #409 (online mode), #422
(multiple LoRAs/Neo). None is claimed to be completely resolved by this change.
