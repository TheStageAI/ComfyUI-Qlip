import logging
from pathlib import Path

import torch

from .utils import (
    find_engines_dir, has_engine_files, _add_qlip_to_path,
    _infer_lora_config_from_model, _discover_block_groups,
    load_lora_config_json
)

logger = logging.getLogger("qlip_nodes")

MAX_LORA_RANK = 256


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
    Load pre-compiled TRT engines and replace transformer blocks
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

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
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
                    "tooltip": "Enable CUDA Graph capture for TRT engines. "
                               "Reduces kernel launch overhead for faster inference. "
                               "First run captures the graph, subsequent runs replay it.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_engines"
    CATEGORY = "qlip"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load_engines(self, model, engines_path="",
                     hf_repo="", lora_stack=None, cuda_graph=False):
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

            self._apply_model_patches(dm)

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

        # Load TRT engines
        if not engines_cached:
            print(f"[qlip] Loading TRT engines from {engines_dir}...")

            from qlip.inference.nvidia import NvidiaInferenceManager
            from qlip.inference.nvidia.session import NvidiaMemoryManager

            imanager = NvidiaInferenceManager(model=dm, workspace=engines_dir)
            imanager.auto_setup()

            # Create memory manager, register sessions
            # (allocate_memory is deferred until AFTER lora wrapper setup)
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

        # Model-specific patches (AFTER engine loading)
        self._apply_model_patches(dm)

        # LoRA wrapper (AFTER engine loading — wraps CompiledModules)
        if with_lora:
            from elastic_models.diffusers.lora import QlipLoraModule

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

        # Allocate device memory AFTER all setup (lora wrapper, patches)
        # This must come last — matches working WAN pattern
        mm.extract_device_memory_size()
        mm.allocate_memory()
        print("[qlip] Device memory allocated")

        # Enable CUDA Graph on TRT sessions
        if cuda_graph:
            self._enable_cuda_graph(imanager)

        return (patched_model,)

    # ------------------------------------------------------------------
    # LoRA cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _disable_lora(groups):
        """Disable LoRA by replacing packed tensors with rank-1 zero tensors.

        Uses rank=1 so TRT selects the smallest optimization profile,
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
        """Hot-swap LoRA weights from a multi-file lora_stack. (use elastic_models.swap_lora in the future)"""
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
        from elastic_models.diffusers.lora import LoRAManager

        has_real_lora = lora_stack and len(lora_stack) > 0

        # --- Resolve configs ---
        if lora_config is not None:
            configs = lora_config
        elif has_real_lora:
            first_path = lora_stack[0]["path"]
            configs = LoRAManager.infer_config(first_path)
            logger.warning(
                "No lora_config.json in engines dir — config inferred from %s",
                Path(first_path).name,
            )
            print(f"[qlip] WARNING: lora_config.json not found in engines dir, "
                  f"config inferred from {Path(first_path).name}")
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
        from elastic_models.diffusers.lora import (
            LoRAManager, LoRABlockGroup, create_zero_lora_packed,
        )

        blocks = getattr(dm, block_attr)
        num_blocks = len(blocks)

        if lora_stack:
            manager = LoRAManager(config, device="cuda", dtype=torch.bfloat16)
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
            used_rank = max_rank
            manager = LoRAManager(config, device="cuda", dtype=torch.bfloat16)
            dummy = create_zero_lora_packed(
                config, used_rank, device="cuda", dtype=torch.bfloat16
            )
            packed = [dummy for _ in range(num_blocks)]
            print(f"[qlip] {block_attr}: {num_blocks} blocks, "
                  f"zero lora_packed (rank={max_rank})")

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
    def _apply_model_patches(dm):
        """Apply model-specific patches after engine loading.

        Detects model type and applies appropriate patches:
        - FLUX Klein: flatten ModulationOut → stacked tensor
        - LTXAV: expand CompressedTimestep + stack pe tuples → tensors
        """
        # FLUX Klein global_modulation
        if (getattr(dm, 'params', None)
                and getattr(dm.params, 'global_modulation', False)):
            from .utils import patch_forward_orig_for_modulation
            patch_forward_orig_for_modulation(dm)

        # LTXAV (audio-video LTX-2)
        from .utils import is_ltxav_model
        if is_ltxav_model(dm):
            from .utils import (
                patch_compressed_timestep,
                patch_process_transformer_blocks,
            )
            patch_compressed_timestep(dm)
            patch_process_transformer_blocks(dm)

    @staticmethod
    def _enable_cuda_graph(imanager):
        """Enable CUDA Graph capture on all TRT sessions.

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
    Enable or disable LoRA on a model with pre-loaded TRT engines.

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
