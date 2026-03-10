import time
import threading

import torch


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
    _cold_starts: dict[str, float] = {}     # name -> first elapsed (persistent)

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
            # Record cold start (first ever measurement for this name)
            if name not in cls._cold_starts:
                cls._cold_starts[name] = elapsed
            return elapsed

    @classmethod
    def get_results(cls) -> list[tuple[str, float]]:
        with cls._lock:
            cls._collected = True  # mark as collected → next start() will reset
            return list(cls._results)

    @classmethod
    def get_cold_start(cls, name: str) -> float | None:
        with cls._lock:
            return cls._cold_starts.get(name)



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
            "optional": {
                "cuda_sync": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Call torch.cuda.synchronize() for accurate GPU timing",
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

    def start_timer(self, passthrough, timer_name="timer_1", cuda_sync=True):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
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
            "optional": {
                "cuda_sync": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Call torch.cuda.synchronize() for accurate GPU timing",
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

    def stop_timer(self, passthrough, timer_name="timer_1", cuda_sync=True):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = _TimerStore.stop(timer_name)
        if elapsed is not None:
            ms = elapsed * 1000
            text = f"{timer_name}: {elapsed:.3f} s ({ms:.1f} ms)"
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
                "track_cold_start": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show cold start (first run) comparison in report",
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

    def report(self, trigger=None, track_cold_start=False):
        results = _TimerStore.get_results()

        if not results:
            text = "No timer data. Add QlipTimerStart/Stop nodes."
            print(f"[qlip timer] {text}")
            return {"ui": {"text": (text,)}}

        lines = ["=== Qlip Timer Report ==="]
        total = 0.0
        cold_total = 0.0
        has_cold = False

        for name, elapsed in results:
            ms = elapsed * 1000
            total += elapsed

            if track_cold_start:
                cold = _TimerStore.get_cold_start(name)
                if cold is not None:
                    cold_ms = cold * 1000
                    cold_total += cold
                    has_cold = True
                    if abs(cold - elapsed) < 1e-6:
                        lines.append(f"  {name}: {elapsed:.3f} s ({ms:.1f} ms)  (cold start)")
                    else:
                        delta_pct = ((elapsed - cold) / cold) * 100
                        lines.append(
                            f"  {name}: {elapsed:.3f} s ({ms:.1f} ms)  "
                            f"(cold: {cold:.3f} s, {delta_pct:+.1f}%)"
                        )
                else:
                    lines.append(f"  {name}: {elapsed:.3f} s ({ms:.1f} ms)")
            else:
                lines.append(f"  {name}: {elapsed:.3f} s ({ms:.1f} ms)")

        lines.append(f"  --------")
        total_ms = total * 1000
        if track_cold_start and has_cold:
            if abs(cold_total - total) < 1e-6:
                lines.append(
                    f"  Total: {total:.3f} s ({total_ms:.1f} ms)  (cold start)"
                )
            else:
                delta_pct = ((total - cold_total) / cold_total) * 100
                lines.append(
                    f"  Total: {total:.3f} s ({total_ms:.1f} ms)  "
                    f"(cold: {cold_total:.3f} s, {delta_pct:+.1f}%)"
                )
        else:
            lines.append(f"  Total: {total:.3f} s ({total_ms:.1f} ms)")
        lines.append("=========================")

        text = "\n".join(lines)
        print(f"[qlip timer]\n{text}")

        return {"ui": {"text": (text,)}}
