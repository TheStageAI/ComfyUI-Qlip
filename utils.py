import sys
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger("qlip_nodes")


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------

def find_engines_dir(engines_path: str = "", hf_repo: str = "") -> Path:
    """
    Resolve engine directory from absolute path or HuggingFace repo.

    engines_path takes priority over hf_repo.

    Args:
        engines_path: Absolute path to directory with .engine / .qlip files
        hf_repo: HuggingFace repo id containing compiled engines

    Returns:
        Path to engines directory

    Raises:
        ValueError: If neither engines_path nor hf_repo is provided
        FileNotFoundError: If engines_path doesn't exist
    """
    if engines_path:
        p = Path(engines_path)
        if not p.is_dir():
            raise FileNotFoundError(f"Engines directory not found: {engines_path}")
        return p
    if hf_repo:
        return download_engines_from_hf(hf_repo)
    raise ValueError(
        "Specify either engines_path (absolute path) or hf_repo (HuggingFace repo)"
    )


def download_engines_from_hf(hf_repo: str, local_dir: Optional[str] = None) -> Path:
    """
    Download compiled engines from a HuggingFace repository.

    Downloads .engine, .qlip, .bin (encryption key), and .json files.
    Uses huggingface_hub cache by default.

    hf_repo can be:
      - "TheStageAI/Elastic-FLUX-2-Klein" — downloads all, finds engines automatically
      - "TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-4b-fp8_lora" — downloads
        only engines from the specified subdirectory

    Args:
        hf_repo: HuggingFace repo id, optionally with :subpath suffix
        local_dir: Optional local directory to save files

    Returns:
        Path to directory containing engine files
    """
    from huggingface_hub import snapshot_download

    # Parse optional subpath: "org/repo:subpath"
    if ":" in hf_repo:
        repo_id, subpath = hf_repo.rsplit(":", 1)
        subpath = subpath.strip("/")
        allow_patterns = [f"{subpath}/*"]
    else:
        repo_id = hf_repo
        subpath = None
        allow_patterns = ["*.engine", "*.qlip", "*.bin", "*.json"]

    path = Path(snapshot_download(
        repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    ))

    # If subpath was specified, return it directly
    if subpath:
        engines_path = path / subpath
        if engines_path.is_dir():
            return engines_path

    # Otherwise find the deepest directory with engine files
    if not _has_engine_files_flat(path):
        for engine_file in path.rglob("*.qlip"):
            return engine_file.parent
        for engine_file in path.rglob("*.engine"):
            return engine_file.parent

    return path


def has_engine_files(engines_dir: Path) -> bool:
    """Check if directory contains .engine or .qlip files (recursively)."""
    if not engines_dir.is_dir():
        return False
    for ext in ("*.engine", "*.qlip"):
        if list(engines_dir.rglob(ext)):
            return True
    return False


def _has_engine_files_flat(engines_dir: Path) -> bool:
    """Check if directory contains .engine or .qlip files (non-recursive)."""
    for ext in ("*.engine", "*.qlip"):
        if list(engines_dir.glob(ext)):
            return True
    return False


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _add_qlip_to_path():
    """Ensure qlip package is importable."""
    qlip_root = Path(__file__).resolve().parents[2]
    if str(qlip_root) not in sys.path:
        sys.path.insert(0, str(qlip_root))


# ---------------------------------------------------------------------------
# LoRA config helpers
# ---------------------------------------------------------------------------

def _infer_lora_config_from_model(dm, block_attr="double_blocks", prefix="double_blocks"):
    """Build LoRAConfig from nn.Linear modules found in block[0].

    Used when engines were compiled with LoRA but no LoRA file is provided
    (zero-tensor case). The config must match the structure used at compilation.

    Args:
        dm: diffusion model (transformer)
        block_attr: attribute name for blocks (e.g. "double_blocks", "single_blocks")
        prefix: block_prefix for LoRAConfig

    Returns:
        LoRAConfig or None if no blocks/linears found
    """
    _add_qlip_to_path()
    from elastic_models.diffusers.lora import LayerConfig, LoRAConfig

    blocks = getattr(dm, block_attr, [])
    if not blocks:
        return None

    block = blocks[0]
    layers = []
    max_features = 0
    for name, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            layers.append(LayerConfig(name, mod.out_features, mod.in_features))
            max_features = max(max_features, mod.out_features, mod.in_features)

    if not layers:
        return None

    return LoRAConfig(
        name=f"auto_{prefix}",
        layers=layers,
        block_prefix=prefix,
        num_blocks=len(blocks),
        max_features=max_features,
    )


def _discover_block_groups(dm):
    """Auto-discover block groups in diffusion model.

    Scans top-level children for ModuleList attributes whose elements
    contain nn.Linear layers (i.e. likely transformer blocks with LoRA targets).

    Works for any model architecture:
    - FLUX: finds double_blocks, single_blocks
    - WAN: finds blocks
    - Future models: any ModuleList with Linear layers

    Returns:
        list of (attr_name, block_prefix) tuples
    """
    groups = []
    for name, module in dm.named_children():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            block = module[0]
            has_linear = any(isinstance(m, torch.nn.Linear)
                             for m in block.modules())
            if has_linear:
                groups.append((name, name))
    return groups


def load_lora_config_json(path):
    """Load list of LoRAConfig from JSON file.

    JSON format::

        {
          "configs": [
            {
              "name": "...",
              "block_prefix": "double_blocks",
              "num_blocks": 24,
              "max_features": 12288,
              "layers": [
                {"name": "img_attn.qkv", "out_features": 9216, "in_features": 3072},
                ...
              ]
            },
            ...
          ]
        }

    Args:
        path: Path to JSON file.

    Returns:
        list[LoRAConfig]
    """
    import json

    _add_qlip_to_path()
    from elastic_models.diffusers.lora import LayerConfig, LoRAConfig

    with open(path) as f:
        data = json.load(f)

    configs = []
    for entry in data["configs"]:
        layers = [
            LayerConfig(l["name"], l["out_features"], l["in_features"])
            for l in entry["layers"]
        ]
        configs.append(LoRAConfig(
            name=entry["name"],
            layers=layers,
            block_prefix=entry["block_prefix"],
            num_blocks=entry["num_blocks"],
            max_features=entry["max_features"],
        ))
    return configs

# ---------------------------------------------------------------------------
# FLUX SPECIFIC — modulation flattening
# ---------------------------------------------------------------------------


def patch_forward_orig_for_modulation(transformer):
    """Monkey-patch Flux.forward_orig to flatten ModulationOut before calling blocks.

    Must be called AFTER setup_modules() — patches the caller, not the blocks.
    Also needed at inference time (nodes.py).
    """
    if not transformer.params.global_modulation:
        return

    orig_forward_orig = transformer.forward_orig

    def patched_forward_orig(
        img, img_ids, txt, txt_ids, timesteps, y,
        guidance=None, control=None, transformer_options={}, attn_mask=None,
    ):
        import comfy.ldm.flux.layers as flux_layers

        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = transformer.img_in(img)
        vec = transformer.time_in(flux_layers.timestep_embedding(timesteps, 256).to(img.dtype))
        if transformer.params.guidance_embed:
            if guidance is not None:
                vec = vec + transformer.guidance_in(flux_layers.timestep_embedding(guidance, 256).to(img.dtype))

        if transformer.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], transformer.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + transformer.vector_in(y[:, :transformer.params.vec_in_dim])

        if transformer.txt_norm is not None:
            txt = transformer.txt_norm(txt)
        txt = transformer.txt_in(txt)

        vec_orig = vec

        # Compute modulation and stack into single tensor (12, batch, 1, hidden)
        img_mod = transformer.double_stream_modulation_img(vec_orig)
        txt_mod = transformer.double_stream_modulation_txt(vec_orig)
        img_mod1, img_mod2 = img_mod
        txt_mod1, txt_mod2 = txt_mod
        stacked_vec = torch.stack([
            img_mod1.shift, img_mod1.scale, img_mod1.gate,
            img_mod2.shift, img_mod2.scale, img_mod2.gate,
            txt_mod1.shift, txt_mod1.scale, txt_mod1.gate,
            txt_mod2.shift, txt_mod2.scale, txt_mod2.gate,
        ])

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = transformer.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(transformer.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(transformer.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"),
                                                   transformer_options=args.get("transformer_options"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": stacked_vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=stacked_vec,
                                 pe=pe,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, :add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        # Single blocks: stack ModulationOut into tensor (3, batch, 1, hidden)
        single_vec_full, _ = transformer.single_stream_modulation(vec_orig)
        stacked_single_vec = torch.stack([single_vec_full.shift, single_vec_full.scale, single_vec_full.gate])

        transformer_options["total_blocks"] = len(transformer.single_blocks)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(transformer.single_blocks):
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"),
                                       transformer_options=args.get("transformer_options"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": stacked_single_vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=stacked_single_vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = transformer.final_layer(img, vec_orig)
        return img

    transformer.forward_orig = patched_forward_orig
    print("  Patched forward_orig: flatten ModulationOut → tuple[Tensor]")


# ---------------------------------------------------------------------------
# LTXAV SPECIFIC — CompressedTimestep expansion + pe stacking
# ---------------------------------------------------------------------------


def is_ltxav_model(dm) -> bool:
    """Check if diffusion model is LTXAV (audio-video LTX-2)."""
    cls_name = type(dm).__name__
    if cls_name == "LTXAVModel":
        return True
    if hasattr(dm, "transformer_blocks") and len(dm.transformer_blocks) > 0:
        block_cls = type(dm.transformer_blocks[0]).__name__
        if block_cls == "BasicAVTransformerBlock":
            return True
    return False


def patch_compressed_timestep(transformer):
    """Patch _prepare_timestep to return raw tensors instead of CompressedTimestep.

    CompressedTimestep is a Python object (not a tensor) — TRT engines expect
    raw tensors. The block's get_ada_values handles plain tensors via the else
    branch, so raw tensors work fine.
    """
    orig_prepare = transformer._prepare_timestep

    def patched_prepare_timestep(timestep, batch_size, hidden_dtype, **kwargs):
        result = orig_prepare(timestep, batch_size, hidden_dtype, **kwargs)
        timesteps, embedded = result

        from comfy.ldm.lightricks.av_model import CompressedTimestep

        def expand_if_compressed(obj):
            if isinstance(obj, CompressedTimestep):
                return obj.expand()
            return obj

        timesteps_raw = []
        for item in timesteps:
            if isinstance(item, list):
                timesteps_raw.append([expand_if_compressed(x) for x in item])
            else:
                timesteps_raw.append(expand_if_compressed(item))

        embedded_raw = [expand_if_compressed(e) for e in embedded]
        return timesteps_raw, embedded_raw

    transformer._prepare_timestep = patched_prepare_timestep
    print("[qlip] Patched _prepare_timestep: CompressedTimestep → raw tensors")


def patch_process_transformer_blocks(transformer):
    """Patch _process_transformer_blocks to stack pe tuples into [2,...] tensors.

    At inference time, TRT engines expect stacked pe tensors (baked at compile).
    The original _process_transformer_blocks passes (cos, sin, split_mode) tuples
    which the engine can't accept. This patch stacks cos+sin and stores split_mode
    on block.model attributes.

    Must be called AFTER engine loading (auto_setup).
    """
    import types

    def patched_process_transformer_blocks(self_tr, x, context, attention_mask,
                                           timestep, pe, transformer_options={},
                                           **kwargs):
        (v_pe_tuple, v_cross_tuple) = pe[0]
        (a_pe_tuple, a_cross_tuple) = pe[1]

        def stack_and_store(pe_tuple):
            cos, sin, split_mode = pe_tuple
            return torch.stack([cos, sin]), split_mode

        v_pe_stacked, v_split = stack_and_store(v_pe_tuple)
        v_cross_stacked, v_cross_split = stack_and_store(v_cross_tuple)
        a_pe_stacked, a_split = stack_and_store(a_pe_tuple)
        a_cross_stacked, a_cross_split = stack_and_store(a_cross_tuple)

        # Store split_mode on underlying blocks (CompiledModule.model)
        for block in self_tr.transformer_blocks:
            blk = block.model if hasattr(block, "model") else block
            blk._v_split_pe = v_split
            blk._a_split_pe = a_split
            blk._v_cross_split_pe = v_cross_split
            blk._a_cross_split_pe = a_cross_split

        vx = x[0]
        ax = x[1]
        v_context_t = context[0]
        a_context_t = context[1]
        v_timestep_t = timestep[0]
        a_timestep_t = timestep[1]
        (
            av_ca_audio_scale_shift_timestep,
            av_ca_video_scale_shift_timestep,
            av_ca_a2v_gate_noise_timestep,
            av_ca_v2a_gate_noise_timestep,
        ) = timestep[2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        for i, block in enumerate(self_tr.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"],
                        v_context=args["v_context"],
                        a_context=args["a_context"],
                        attention_mask=args["attention_mask"],
                        v_timestep=args["v_timestep"],
                        a_timestep=args["a_timestep"],
                        v_pe=args["v_pe"],
                        a_pe=args["a_pe"],
                        v_cross_pe=args["v_cross_pe"],
                        a_cross_pe=args["a_cross_pe"],
                        v_cross_scale_shift_timestep=args["v_cross_scale_shift_timestep"],
                        a_cross_scale_shift_timestep=args["a_cross_scale_shift_timestep"],
                        v_cross_gate_timestep=args["v_cross_gate_timestep"],
                        a_cross_gate_timestep=args["a_cross_gate_timestep"],
                        transformer_options=args["transformer_options"],
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {
                        "img": (vx, ax),
                        "v_context": v_context_t,
                        "a_context": a_context_t,
                        "attention_mask": attention_mask,
                        "v_timestep": v_timestep_t,
                        "a_timestep": a_timestep_t,
                        "v_pe": v_pe_stacked,
                        "a_pe": a_pe_stacked,
                        "v_cross_pe": v_cross_stacked,
                        "a_cross_pe": a_cross_stacked,
                        "v_cross_scale_shift_timestep": av_ca_video_scale_shift_timestep,
                        "a_cross_scale_shift_timestep": av_ca_audio_scale_shift_timestep,
                        "v_cross_gate_timestep": av_ca_a2v_gate_noise_timestep,
                        "a_cross_gate_timestep": av_ca_v2a_gate_noise_timestep,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                vx, ax = out["img"]
            else:
                vx, ax = block(
                    (vx, ax),
                    v_context=v_context_t,
                    a_context=a_context_t,
                    attention_mask=attention_mask,
                    v_timestep=v_timestep_t,
                    a_timestep=a_timestep_t,
                    v_pe=v_pe_stacked,
                    a_pe=a_pe_stacked,
                    v_cross_pe=v_cross_stacked,
                    a_cross_pe=a_cross_stacked,
                    v_cross_scale_shift_timestep=av_ca_video_scale_shift_timestep,
                    a_cross_scale_shift_timestep=av_ca_audio_scale_shift_timestep,
                    v_cross_gate_timestep=av_ca_a2v_gate_noise_timestep,
                    a_cross_gate_timestep=av_ca_v2a_gate_noise_timestep,
                    transformer_options=transformer_options,
                )

        return [vx, ax]

    transformer._process_transformer_blocks = types.MethodType(
        patched_process_transformer_blocks, transformer
    )
    print("[qlip] Patched _process_transformer_blocks: stack pe → [2,...] tensors")
