import logging
from pathlib import Path

import torch

from ..utils import (
    find_engines_dir, has_engine_files, _add_qlip_to_path,
    convert_lora_format,
    _infer_lora_config_from_model, _discover_block_groups,
    load_lora_config_json
)

logger = logging.getLogger("qlip_nodes")

MAX_LORA_RANK = 256

# Cache for custom patch modules (keyed by engines_dir path)
_custom_patch_cache = {}


def _get_custom_patch_module(engines_dir):
    """Load and cache the qlip_patch.py module from engines directory."""
    import importlib.util

    engines_dir = str(engines_dir)
    patch_file = Path(engines_dir) / "qlip_patch.py"
    if not patch_file.is_file():
        return None

    if engines_dir not in _custom_patch_cache:
        spec = importlib.util.spec_from_file_location(
            f"qlip_patch_{hash(engines_dir)}", str(patch_file)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _custom_patch_cache[engines_dir] = mod
        logger.info(f"Loaded custom patch module from {patch_file}")

    return _custom_patch_cache[engines_dir]


def _has_custom_patch(engines_dir, func_name):
    """Return True if qlip_patch.py defines `func_name`. Custom patches override built-in."""
    if not engines_dir:
        return False
    mod = _get_custom_patch_module(engines_dir)
    return mod is not None and getattr(mod, func_name, None) is not None


def _load_custom_patch(engines_dir, func_name, dm):
    """Load and call a function from qlip_patch.py in the engines directory.

    Supports two hook functions:
        patch_signatures(dm) — called BEFORE auto_setup()
        patch_caller(dm)     — called AFTER engine loading
    """
    if not engines_dir:
        return
    mod = _get_custom_patch_module(engines_dir)
    if mod is None:
        return
    fn = getattr(mod, func_name, None)
    if fn is not None:
        fn(dm)
        logger.info(f"Applied custom {func_name}() from "
                    f"{Path(engines_dir) / 'qlip_patch.py'}")


class QlipLoraStack:
    """
    Build a chainable list of LoRA entries.

    Each node adds one (path, strength) entry. Chain multiple nodes
    via the prev_stack input to stack LoRAs (ranks are concatenated).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to LoRA safetensors file",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "LoRA strength multiplier",
                }),
            },
            "optional": {
                "prev_stack": ("QLIP_LORA_STACK", {
                    "tooltip": "Previous LoRA stack to extend (chain multiple LoRAs)",
                }),
            },
        }

    RETURN_TYPES = ("QLIP_LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack_lora"
    CATEGORY = "qlip"

    def stack_lora(self, lora_path, strength=1.0, prev_stack=None):
        stack = list(prev_stack) if prev_stack else []

        if not lora_path:
            return (stack,)

        resolved = str(Path(lora_path).resolve())
        if not Path(resolved).exists():
            raise FileNotFoundError(f"LoRA file not found: {resolved}")

        stack.append({"path": resolved, "strength": strength})
        return (stack,)


class QlipEnginesLoader:
    """
    Load pre-compiled QLIP engines and replace transformer blocks
    with CompiledModule instances.

    LoRA support is auto-detected:
    - If lora_config.json exists in the engines directory → LoRA-enabled engines
    - If lora_stack is connected → LoRA is needed (config from json or inferred)
    - Otherwise → no LoRA

    Caches engines and LoRA state across invocations:
    - Engines are loaded once and reused (keyed by engines_dir).
    - LoRA weights are hot-swapped via swap_lora when lora_stack changes.
    - If lora_stack is unchanged between runs, no work is done.
    """

    _engines_cache = {}      # str(engines_dir) → (imanager, memory_manager)
    _lora_groups_cache = {}  # str(engines_dir) → list[LoRABlockGroup]
    _last_lora_key = {}      # str(engines_dir) → tuple (hash of lora_stack)
    _lora_supported = {}     # str(engines_dir) → bool (engines have LoRA support)
    _shared_mm = {}          # str(group_name) → NvidiaMemoryManager (shared across loaders)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("*",),
            },
            "optional": {
                "engines_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to directory with .qlip/.engine files",
                }),
                "hf_repo": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace repo with engines, e.g. "
                               "TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-9b-fp8_lora",
                }),
                "lora_stack": ("QLIP_LORA_STACK", {
                    "tooltip": "LoRA stack from QlipLoraStack node(s)",
                }),
                "cuda_graph": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable CUDA Graph capture for QLIP engines. "
                               "Reduces kernel launch overhead for faster inference. "
                               "First run captures the graph, subsequent runs replay it.",
                }),
                "shared_memory": ("STRING", {
                    "default": "",
                    "tooltip": "Shared memory group name. Multiple QlipEnginesLoader nodes "
                               "with the same name share one GPU memory pool (size = max "
                               "across all sessions, not sum). Each loader deallocates → "
                               "re-allocates, so every transformer works immediately. "
                               "Useful for WAN 2.2 (high and low transformers).",
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_engines"
    CATEGORY = "qlip"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load_engines(self, model, engines_path="",
                     hf_repo="", lora_stack=None, cuda_graph=False,
                     shared_memory=""):
        if not engines_path and not hf_repo:
            print("[qlip] No engines_path or hf_repo specified, "
                  "passing model through unchanged")
            return (model,)

        _add_qlip_to_path()

        # Resolve engines directory
        engines_dir = find_engines_dir(engines_path, hf_repo)
        cache_key = str(engines_dir)

        if not has_engine_files(engines_dir):
            raise FileNotFoundError(
                f"No .qlip/.engine files found in {engines_dir}. "
                f"Compile engines first or download precompiled ones."
            )

        # Auto-detect LoRA support from lora_config.json or lora_stack
        lora_config_path = Path(engines_dir) / "lora_config.json"
        has_lora_config = lora_config_path.is_file()
        has_lora_stack = lora_stack and len(lora_stack) > 0
        with_lora = has_lora_config or has_lora_stack

        # Load lora_config.json if present
        lora_config = None
        if has_lora_config:
            lora_config = load_lora_config_json(str(lora_config_path))
            print(f"[qlip] Loaded lora_config.json: "
                  f"{len(lora_config)} block group(s)")

        # Clone the model patcher
        patched_model = model.clone()
        dm = patched_model.model.diffusion_model


        engines_cached = cache_key in self._engines_cache
        lora_key = self._compute_lora_key(lora_stack)

        # ==============================================================
        # Fast path: engines + LoRA already cached → swap or skip
        # ==============================================================
        if engines_cached and with_lora and cache_key in self._lora_groups_cache:
            imanager, mm = self._engines_cache[cache_key]
            dm._qlip_imanager = imanager
            dm._qlip_memory_manager = mm

            self._apply_signature_patches(dm, engines_dir)
            self._apply_caller_patches(dm, engines_dir)

            groups = self._lora_groups_cache[cache_key]
            prev_key = self._last_lora_key.get(cache_key)

            if lora_key == prev_key:
                # Same LoRA — nothing to do
                print("[qlip] LoRA unchanged, skipping swap")
            elif lora_key is None:
                # LoRA removed — rank-1 zero for minimal overhead
                self._disable_lora(groups)
                print("[qlip] LoRA disabled (rank=1)")
            else:
                # Different LoRA — hot-swap
                self._swap_lora_stack(groups, lora_stack, MAX_LORA_RANK)

            self._last_lora_key[cache_key] = lora_key

            dm._qlip_lora_groups = groups
            print(f"[qlip] Using cached engines from {engines_dir}")
            return (patched_model,)

        # ==============================================================
        # Full path: first load
        # ==============================================================

        # LoRA setup (BEFORE engine loading — signatures must be ready)
        lora_groups = []
        if with_lora:
            lora_groups = self._setup_lora(
                dm, lora_stack, lora_config, MAX_LORA_RANK
            )

        # Signature patches BEFORE engine loading —
        # auto_setup() reads block.model.forward signature for input mapping
        self._apply_signature_patches(dm, engines_dir)

        # Load QLIP engines
        if not engines_cached:
            print(f"[qlip] Loading QLIP engines from {engines_dir}...")

            from qlip.inference.nvidia import NvidiaInferenceManager
            from qlip.inference.nvidia.session import NvidiaMemoryManager

            imanager = NvidiaInferenceManager(model=dm, workspace=engines_dir)
            imanager.auto_setup()

            # Create or reuse memory manager.
            # When shared_memory is set, all loaders in the same group
            # register their sessions into one NvidiaMemoryManager.
            # Pool size = max(all sessions), not sum — so two transformers
            # that never run simultaneously share one pool.
            if shared_memory:
                if shared_memory not in self._shared_mm:
                    self._shared_mm[shared_memory] = NvidiaMemoryManager()
                    print(f"[qlip] Created shared memory group "
                          f"'{shared_memory}'")
                mm = self._shared_mm[shared_memory]
            else:
                mm = NvidiaMemoryManager()

            for mod in imanager.modules:
                mm.add_infsession(mod.session)

            self._engines_cache[cache_key] = (imanager, mm)
            self._lora_supported[cache_key] = with_lora
            print(f"[qlip] Loaded {len(imanager.modules)} engine modules")
        else:
            imanager, mm = self._engines_cache[cache_key]
            print(f"[qlip] Using cached engines from {engines_dir}")

        dm._qlip_imanager = imanager
        dm._qlip_memory_manager = mm

        # Caller patches AFTER engine loading
        self._apply_caller_patches(dm, engines_dir)

        # LoRA wrapper (AFTER engine loading — wraps CompiledModules)
        if with_lora:
            from qlip.lora_support import QlipLoraModule

            for group in lora_groups:
                QlipLoraModule.setup(
                    dm, group.block_prefix, group.config, group.packed,
                )
                print(f"[qlip] LoRA wrapper: "
                      f"{group.num_blocks} {group.block_prefix}")

            dm._qlip_lora_groups = lora_groups

            # Cache LoRA state for future swaps
            self._lora_groups_cache[cache_key] = lora_groups
            self._last_lora_key[cache_key] = lora_key

        # Allocate device memory AFTER all setup (lora wrapper, patches).
        #
        # For shared_memory groups: the mm may already be allocated from a
        # previous loader in the group. We must deallocate first, then
        # re-extract sizes (now covering sessions from ALL loaders so far),
        # then re-allocate. This is safe because:
        #   - The previous transformer already finished its sampler pass
        #     (ComfyUI executes nodes in dependency order)
        #   - allocate_memory() sets the new device_memory pointer on ALL
        #     sessions in the mm (including the previous transformer's),
        #     so the previous transformer will also work on subsequent runs
        #   - Pool size = max(all sessions), so it doesn't grow linearly
        if shared_memory and hasattr(mm, "device_mem"):
            mm.deallocate_memory()
            print(f"[qlip] Deallocated shared memory group "
                  f"'{shared_memory}' for re-allocation")

        mm.extract_device_memory_size()
        mm.allocate_memory()

        if shared_memory:
            print(f"[qlip] Shared memory allocated for group "
                  f"'{shared_memory}' "
                  f"({len(mm._infsessions)} sessions)")
        else:
            print("[qlip] Device memory allocated")

        # Enable CUDA Graph on QLIP sessions
        if cuda_graph:
            self._enable_cuda_graph(imanager)

        # Disable custom loader features that conflict with compiled engines
        # (load_weights, auto_cpu_offload, block_swap). Works for any loader
        # type — WanVideoWrapper, LTX custom nodes, etc.
        self._disable_custom_loader_features(dm, patched_model)

        return (patched_model,)

    @staticmethod
    def _disable_custom_loader_features(dm, model_patcher):
        """Disable weight loading/offloading from custom loaders and fix dtypes.

        Custom loaders (WanVideoWrapper, etc.) re-load weights and offload
        the model between runs. With compiled engines this is unnecessary
        and breaks CompiledModule blocks.

        Also fixes dtype mismatch: comfy.sd may load BF16 checkpoints as FP16,
        but engines are compiled with BF16 inputs. Convert non-compiled parts
        of the model to BF16 so activations flowing into compiled blocks
        are BF16.

        Fields set:
        - dm.patched_linear = True → skip _replace_linear, use meta offload path
        - model["sd"] = None → skip load_weights (guarded by sd is not None)
        - model["auto_cpu_offload"] = False → skip CPU offloading
        """
        # WanVideoWrapper: skip _replace_linear
        dm.patched_linear = True

        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, '__setitem__'):
            try:
                # Clear state_dict → load_weights guard: "if sd is not None" → skip
                model_patcher.model["sd"] = None
                # Disable auto CPU offload
                model_patcher.model["auto_cpu_offload"] = False
            except (KeyError, TypeError):
                pass
    # ------------------------------------------------------------------
    # LoRA cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _disable_lora(groups):
        """Disable LoRA by replacing packed tensors with rank-1 zero tensors.

        Uses rank=1 so QLIP selects the smallest optimization profile,
        minimizing LoRA MatMul overhead when LoRA is not active.
        """
        for g in groups:
            for i, packed in enumerate(g.packed):
                g.packed[i] = torch.zeros(
                    packed.shape[0], 1, packed.shape[2], packed.shape[3],
                    device=packed.device, dtype=packed.dtype,
                )

    @staticmethod
    def _compute_lora_key(lora_stack):
        """Compute a hashable key from lora_stack for cache comparison.

        Returns None if no LoRA, or tuple of (path, strength) pairs.
        """
        if not lora_stack or len(lora_stack) == 0:
            return None
        return tuple((e["path"], e["strength"]) for e in lora_stack)

    # ------------------------------------------------------------------
    # LoRA hot-swap (multi-file)
    # ------------------------------------------------------------------

    @staticmethod
    def _swap_lora_stack(groups, lora_stack, max_rank):
        """Hot-swap LoRA weights from a multi-file lora_stack."""
        # Clear old weights
        for g in groups:
            g.manager.clear_weights()

        # Load all LoRA files into managers
        for g in groups:
            for entry in lora_stack:
                count = g.manager.load_from_safetensors(
                    entry["path"], entry["strength"]
                )
                logger.info(
                    "[swap] %s: loaded %d layers from %s",
                    g.block_prefix, count, Path(entry["path"]).name,
                )

        # Compute uniform rank across all groups
        used_rank = max(
            g.manager.compute_total_rank(min_rank=1, max_rank=max_rank)
            for g in groups
        )

        # Re-pack each group
        for g in groups:
            new_packed = []
            for i in range(g.num_blocks):
                new_packed.append(
                    g.manager.pack_block(f"{g.block_prefix}.{i}", used_rank)
                )

            if g.packed and g.packed[0].shape[1] == used_rank:
                for i in range(len(g.packed)):
                    g.packed[i].copy_(new_packed[i])
            else:
                g.packed[:] = new_packed

            g.manager.clear_weights()

        print(f"[qlip] LoRA swapped: rank={used_rank}, "
              f"{len(lora_stack)} file(s)")

    # ------------------------------------------------------------------
    # Internal: LoRA first-time setup
    # ------------------------------------------------------------------

    def _setup_lora(self, dm, lora_stack, lora_config, max_rank):
        """Resolve LoRA configs and pack weights into LoRABlockGroups.

        Called BEFORE engine loading to prepare packed tensors.
        QlipLoraModule.setup() is called separately AFTER engine loading
        to wrap CompiledModule blocks.

        Config priority:
        1. lora_config.json from engines directory — exact match
        2. Infer from LoRA file
        3. Infer from model structure

        Returns:
            list[LoRABlockGroup]
        """
        from qlip.lora_support import LoRAManager

        has_real_lora = lora_stack and len(lora_stack) > 0

        # --- Resolve configs ---
        if lora_config is not None:
            configs = lora_config
        elif has_real_lora:
            first_path = lora_stack[0]["path"]
            configs = LoRAManager.infer_config(
                first_path, lora_format_converter=convert_lora_format,
            )
            if configs:
                logger.warning(
                    "No lora_config.json in engines dir — config inferred from %s",
                    Path(first_path).name,
                )
                print(f"[qlip] WARNING: lora_config.json not found in engines dir, "
                      f"config inferred from {Path(first_path).name}")
            else:
                # infer_config() failed — fallback to inferring from model structure.
                # Critical: if engine was compiled with LoRA, blocks expect lora_packed
                # input. Without QlipLoraModule.setup → missing lora_packed error.
                logger.warning(
                    "LoRAManager.infer_config returned [] for %s — "
                    "falling back to model structure inference",
                    Path(first_path).name,
                )
                print(f"[qlip] WARNING: infer_config returned [] for "
                      f"{Path(first_path).name}, falling back to model structure")
                configs = []
                for attr, prefix in _discover_block_groups(dm):
                    cfg = _infer_lora_config_from_model(dm, attr, prefix)
                    if cfg:
                        configs.append(cfg)
        else:
            configs = []
            for attr, prefix in _discover_block_groups(dm):
                cfg = _infer_lora_config_from_model(dm, attr, prefix)
                if cfg:
                    configs.append(cfg)
            logger.warning(
                "No lora_config.json — config inferred from model structure"
            )
            print("[qlip] WARNING: lora_config.json not found, "
                  "config inferred from model structure")

        # --- Setup each block group ---
        lora_groups = []
        for config in configs:
            blocks = getattr(dm, config.block_prefix, None)
            if blocks is None or len(blocks) == 0:
                logger.warning(
                    "No blocks found at dm.%s, skipping",
                    config.block_prefix,
                )
                continue
            group = self._setup_block_group(
                dm, config.block_prefix, config,
                lora_stack if has_real_lora else None,
                max_rank,
            )
            lora_groups.append(group)

        return lora_groups

    def _setup_block_group(self, dm, block_attr, config, lora_stack, max_rank):
        """Setup one block group: pack tensors for LoRA."""
        from qlip.lora_support import (
            LoRAManager, LoRABlockGroup, create_zero_lora_packed,
        )

        blocks = getattr(dm, block_attr)
        num_blocks = len(blocks)

        if lora_stack:
            manager = LoRAManager(config, device="cuda", dtype=torch.bfloat16,
                                    lora_format_converter=convert_lora_format)
            for entry in lora_stack:
                count = manager.load_from_safetensors(
                    entry["path"], entry["strength"]
                )
                print(f"[qlip] Loaded LoRA {Path(entry['path']).name} "
                      f"({count} layers, strength={entry['strength']})")

            used_rank = manager.compute_total_rank(
                min_rank=1, max_rank=max_rank
            )
            packed = []
            for i in range(num_blocks):
                p = manager.pack_block(f"{config.block_prefix}.{i}", used_rank)
                packed.append(p)
            manager.clear_weights()
            print(f"[qlip] {block_attr}: {num_blocks} blocks, rank={used_rank}")
        else:
            used_rank = 1  # Minimal rank for zero LoRA — least compute overhead
            manager = LoRAManager(config, device="cuda", dtype=torch.bfloat16,
                                    lora_format_converter=convert_lora_format)
            dummy = create_zero_lora_packed(
                config, used_rank, device="cuda", dtype=torch.bfloat16
            )
            packed = [dummy for _ in range(num_blocks)]
            print(f"[qlip] {block_attr}: {num_blocks} blocks, "
                  f"zero lora_packed (rank={used_rank})")

        group = LoRABlockGroup(
            manager=manager,
            config=config,
            block_prefix=config.block_prefix,
            num_blocks=num_blocks,
            packed=packed,
        )

        return group

    # ------------------------------------------------------------------
    # Internal: model-specific patches
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_signature_patches(dm, engines_dir=None):
        """Apply block signature patches BEFORE engine loading.

        These patches change block.model.forward signature so that
        auto_setup() sees the correct input names matching the engine.
        Only needed for models where block.forward was restructured
        at compile time (e.g., merged inputs into joint tensor).

        If qlip_patch.py defines patch_signatures(), it overrides built-in patches.
        """
        # Custom patch takes priority — skip built-in if custom is present
        custom_present = _has_custom_patch(engines_dir, "patch_signatures")

        if not custom_present:
            # Qwen Image Edit — block signature was restructured at compile time
            from ..utils import is_qwen_image_model
            if is_qwen_image_model(dm):
                from ..utils import patch_qwen_image_block_signature
                patch_qwen_image_block_signature(dm)

        # Custom patch from engines directory
        if engines_dir:
            _load_custom_patch(engines_dir, "patch_signatures", dm)

    @staticmethod
    def _apply_caller_patches(dm, engines_dir=None):
        """Apply caller patches AFTER engine loading.

        These patches modify the transformer's forward to prepare
        inputs for compiled blocks (concat, stack, set attributes).

        If qlip_patch.py defines patch_caller(), it overrides built-in patches.
        """
        # Custom patch takes priority — skip built-in if custom is present
        custom_present = _has_custom_patch(engines_dir, "patch_caller")

        if not custom_present:
            # FLUX Klein global_modulation
            if (getattr(dm, 'params', None)
                    and getattr(dm.params, 'global_modulation', False)):
                from ..utils import patch_forward_orig_for_modulation
                patch_forward_orig_for_modulation(dm)

            # LTXAV (audio-video LTX-2)
            from ..utils import is_ltxav_model
            if is_ltxav_model(dm):
                from ..utils import (
                    patch_compressed_timestep,
                    patch_process_transformer_blocks,
                )
                patch_compressed_timestep(dm)
                patch_process_transformer_blocks(dm)

            # # Z-Image-Turbo / Lumina2 NextDiT — force fixed cap_feats length
            # from ..utils import is_zimage_lumina_model, patch_zimage_fixed_cap_len
            # if is_zimage_lumina_model(dm):
            #     # Engines are compiled with cap_feats=64. Force runtime to match.
            #     patch_zimage_fixed_cap_len(dm, fixed_cap_len=64)

            # Qwen Image Edit — concat txt+img into joint_hidden_states caller patch
            from ..utils import is_qwen_image_model
            if is_qwen_image_model(dm):
                from ..utils import patch_qwen_image_caller
                patch_qwen_image_caller(dm, fixed_txt_len=1536)

        # Custom patch from engines directory
        if engines_dir:
            _load_custom_patch(engines_dir, "patch_caller", dm)

    @staticmethod
    def _enable_cuda_graph(imanager):
        """Enable CUDA Graph capture on all Qlip sessions.

        Must be called AFTER memory allocation — sessions need
        pre-allocated tensors (store_tensors=True) for graph capture.
        """
        import importlib
        cudart = importlib.import_module("cuda.bindings.runtime")

        count = 0
        for mod in imanager.modules:
            session = mod.session
            session.config.use_cuda_graph = True
            session.config.store_tensors = True
            count += 1
        print(f"[qlip] CUDA Graph enabled on {count} engine sessions")


class QlipLoraSwitch:
    """
    Enable or disable LoRA on a model with pre-loaded QLIP engines.

    Must be used AFTER QlipEnginesLoader with LoRA-enabled engines.
    Mutates LoRA packed tensors in-place — the wrapper sees changes
    automatically (no re-wrapping needed).

    Usage:
    - enable=True + lora_stack connected: swap_lora (load weights)
    - enable=True + lora_stack not connected: keep current state
    - enable=False: disable_lora (replace with rank-1 zero tensors for minimal overhead)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable (swap/keep) or disable (zero) LoRA",
                }),
            },
            "optional": {
                "lora_stack": ("QLIP_LORA_STACK", {
                    "tooltip": "LoRA stack to load (only used when enable=True)",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "switch_lora"
    CATEGORY = "qlip"

    def switch_lora(self, model, enable=True, lora_stack=None):
        patched_model = model.clone()
        dm = patched_model.model.diffusion_model

        groups = getattr(dm, '_qlip_lora_groups', None)
        if groups is None:
            raise RuntimeError(
                "No LoRA groups found on model. "
                "Use LoRA-enabled engines (with lora_config.json) first."
            )

        if not enable:
            QlipEnginesLoader._disable_lora(groups)
            print("[qlip] QlipLoraSwitch: LoRA disabled (rank=1)")
        elif lora_stack and len(lora_stack) > 0:
            QlipEnginesLoader._swap_lora_stack(
                groups, lora_stack, MAX_LORA_RANK
            )
            print("[qlip] QlipLoraSwitch: LoRA swapped")
        else:
            print("[qlip] QlipLoraSwitch: LoRA enabled (keeping current)")

        return (patched_model,)
