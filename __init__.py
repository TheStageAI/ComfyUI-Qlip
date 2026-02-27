from .nodes import QlipEnginesLoader, QlipLoraStack, QlipLoraConfig, QlipLoraSwitch

NODE_CLASS_MAPPINGS = {
    "QlipEnginesLoader": QlipEnginesLoader,
    "QlipLoraStack": QlipLoraStack,
    "QlipLoraConfig": QlipLoraConfig,
    "QlipLoraSwitch": QlipLoraSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QlipEnginesLoader": "Qlip Engines Loader",
    "QlipLoraStack": "Qlip LoRA Stack",
    "QlipLoraConfig": "Qlip LoRA Config",
    "QlipLoraSwitch": "Qlip LoRA Switch",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
