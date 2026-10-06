"""LoRA key conversion for engines (utils/helpers.convert_lora_format)."""

import pytest


@pytest.fixture(scope="module")
def helpers(pack_module):
    return pack_module("utils.helpers")


KREA_BLOCK_KEYS = [
    "transformer_blocks.3.attn.to_q.lora_A.weight",
    "transformer_blocks.3.attn.to_out.0.lora_B.weight",
    "transformer_blocks.3.attn.to_gate.lora_A.weight",
    "transformer_blocks.3.ff.down.lora_A.weight",
]
EXPECTED = {
    "blocks.3.attn.wq.lora_down.weight",
    "blocks.3.attn.wo.lora_up.weight",
    "blocks.3.attn.gate.lora_down.weight",
    "blocks.3.mlp.down.lora_down.weight",
}


@pytest.mark.parametrize(
    "prefix", ["", "transformer.", "diffusion_model.", "model.diffusion_model."]
)
def test_krea2_diffusers_lora_is_remapped(helpers, prefix):
    """Official Comfy-Org Krea 2 LoRAs carry a ``transformer.`` prefix; without
    the remap no key matched and an empty rank-1 LoRA reached the engine
    ("no optimization profile defined")."""
    raw = {prefix + k: i for i, k in enumerate(KREA_BLOCK_KEYS)}
    raw[prefix + "text_fusion.projector.lora_A.weight"] = 99  # not in the engines
    assert helpers._is_krea2_diffusers_lora(raw)
    out = helpers.convert_lora_format(raw)
    assert EXPECTED <= set(out)
    # values are carried over untouched, non-block keys pass through
    assert out["blocks.3.attn.wq.lora_down.weight"] == 0
    assert 99 in out.values()


def test_non_krea_lora_untouched(helpers):
    raw = {"double_blocks.0.img_attn.qkv.lora_A.weight": 1}
    assert not helpers._is_krea2_diffusers_lora(raw)
    assert helpers.convert_lora_format(raw) == raw
