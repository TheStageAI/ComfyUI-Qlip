from .nodes import QlipEnginesLoader, QlipLoraStack, QlipLoraSwitch
from .timer_nodes import QlipTimerStart, QlipTimerStop, QlipTimerReport

NODE_CLASS_MAPPINGS = {
    "QlipEnginesLoader": QlipEnginesLoader,
    "QlipLoraStack": QlipLoraStack,
    "QlipLoraSwitch": QlipLoraSwitch,
    "QlipTimerStart": QlipTimerStart,
    "QlipTimerStop": QlipTimerStop,
    "QlipTimerReport": QlipTimerReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QlipEnginesLoader": "Qlip Engines Loader",
    "QlipLoraStack": "Qlip LoRA Stack",
    "QlipLoraSwitch": "Qlip LoRA Switch",
    "QlipTimerStart": "Qlip Timer Start",
    "QlipTimerStop": "Qlip Timer Stop",
    "QlipTimerReport": "Qlip Timer Report",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
