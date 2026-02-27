import time
import threading


# ---------------------------------------------------------------------------
# ANY type passthrough (SmartType pattern from ComfyUI)
# ---------------------------------------------------------------------------

class _AnyType(str):
    """String subclass that matches any ComfyUI type via __ne__ override."""

    def __ne__(self, other):
        if self == "*" or other == "*":
            return False
        if not isinstance(other, str):
            return True
        return not (set(self.split(",")) & set(other.split(",")))


ANY_TYPE = _AnyType("*")


# ---------------------------------------------------------------------------
# Timer storage (class-level singleton, thread-safe)
# ---------------------------------------------------------------------------

class _TimerStore:
    _lock = threading.Lock()
    _timers: dict[str, float] = {}          # name -> start perf_counter
    _results: list[tuple[str, float]] = []  # ordered (name, elapsed_s)
    _collected = False                       # True after report() reads results

    @classmethod
    def start(cls, name: str):
        with cls._lock:
            # Auto-reset: if results were already collected (by Report),
            # this is a new prompt execution — clear everything.
            if cls._collected:
                cls._timers.clear()
                cls._results.clear()
                cls._collected = False
            cls._timers[name] = time.perf_counter()

    @classmethod
    def stop(cls, name: str) -> float | None:
        now = time.perf_counter()
        with cls._lock:
            start = cls._timers.pop(name, None)
            if start is None:
                return None
            elapsed = now - start
            cls._results.append((name, elapsed))
            return elapsed

    @classmethod
    def get_results(cls) -> list[tuple[str, float]]:
        with cls._lock:
            cls._collected = True  # mark as collected → next start() will reset
            return list(cls._results)


# ---------------------------------------------------------------------------
# QlipTimerStart
# ---------------------------------------------------------------------------

class QlipTimerStart:
    """Record a start timestamp. Place before the node(s) you want to measure."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "passthrough": (ANY_TYPE, {
                    "tooltip": "Data to pass through (any type)",
                }),
                "timer_name": ("STRING", {
                    "default": "timer_1",
                    "tooltip": "Name for this timer (must match QlipTimerStop)",
                }),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "start_timer"
    CATEGORY = "qlip/profiling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def start_timer(self, passthrough, timer_name="timer_1"):
        _TimerStore.start(timer_name)
        print(f"[qlip timer] '{timer_name}' started")
        return (passthrough,)


# ---------------------------------------------------------------------------
# QlipTimerStop
# ---------------------------------------------------------------------------

class QlipTimerStop:
    """Record elapsed time and display it. Place after the measured node(s)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "passthrough": (ANY_TYPE, {
                    "tooltip": "Data to pass through (any type)",
                }),
                "timer_name": ("STRING", {
                    "default": "timer_1",
                    "tooltip": "Name for this timer (must match QlipTimerStart)",
                }),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("passthrough",)
    OUTPUT_NODE = True
    FUNCTION = "stop_timer"
    CATEGORY = "qlip/profiling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def stop_timer(self, passthrough, timer_name="timer_1"):
        elapsed = _TimerStore.stop(timer_name)
        if elapsed is not None:
            ms = elapsed * 1000
            text = f"{timer_name}: {ms:.1f} ms ({elapsed:.3f} s)"
        else:
            text = f"{timer_name}: no matching QlipTimerStart"

        print(f"[qlip timer] {text}")

        return {
            "ui": {"text": (text,)},
            "result": (passthrough,),
        }


# ---------------------------------------------------------------------------
# QlipTimerReport
# ---------------------------------------------------------------------------

class QlipTimerReport:
    """Display a summary table of all timer measurements."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "trigger": (ANY_TYPE, {
                    "tooltip": "Connect any output to ensure execution order",
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "report"
    CATEGORY = "qlip/profiling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def report(self, trigger=None):
        results = _TimerStore.get_results()

        if not results:
            text = "No timer data. Add QlipTimerStart/Stop nodes."
            print(f"[qlip timer] {text}")
            return {"ui": {"text": (text,)}}

        lines = ["=== Qlip Timer Report ==="]
        total = 0.0
        for name, elapsed in results:
            ms = elapsed * 1000
            total += elapsed
            lines.append(f"  {name}: {ms:.1f} ms")
        lines.append(f"  --------")
        lines.append(f"  Total: {total * 1000:.1f} ms ({total:.3f} s)")
        lines.append("=========================")

        text = "\n".join(lines)
        print(f"[qlip timer]\n{text}")

        return {"ui": {"text": (text,)}}
