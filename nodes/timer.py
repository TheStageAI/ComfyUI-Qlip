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

def _gpu_used_bytes() -> int | None:
    """GPU memory currently in use ON THE DEVICE (driver view), in bytes.

    Uses ``torch.cuda.mem_get_info()`` (free, total) → used = total - free.
    This is the DRIVER's view, so it counts memory allocated OUTSIDE the
    PyTorch caching allocator too — crucially the TensorRT engine weights and
    scratch pool, which ``torch.cuda.memory_allocated()`` does NOT see. That's
    why qlip engine VRAM must be measured this way.

    Returns None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return total - free


class _GPUPoller:
    """Background thread that samples total GPU memory in use (driver view)
    every ``interval`` seconds and records the PEAK while running.

    Driver view (mem_get_info → total - free) is used so the peak includes
    memory allocated OUTSIDE PyTorch — notably the TensorRT engine weights and
    per-block scratch pool. That's why a Start/Stop snapshot is not enough: the
    real peak happens MID-interval (e.g. during a sampler forward) and is gone
    by the time Stop runs. The poller catches it.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_used = 0
        self.total = 0

    def _run(self):
        while not self._stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                if used > self.peak_used:
                    self.peak_used = used
                self.total = total
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        # seed with the current reading so a very short interval still has data
        try:
            free, total = torch.cuda.mem_get_info()
            self.peak_used = total - free
            self.total = total
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        # take one final reading, then stop
        try:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            if used > self.peak_used:
                self.peak_used = used
            self.total = total
        except Exception:
            pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_used, self.total


class _TimerStore:
    _lock = threading.Lock()
    _timers: dict[str, float] = {}          # name -> start perf_counter
    _pollers: dict[str, "_GPUPoller"] = {}  # name -> running GPU poller
    # ordered (name, elapsed_s, peak_used_bytes|None, total_bytes|None)
    _results: list[tuple] = []
    _collected = False                       # True after report() reads results
    _cold_starts: dict[str, float] = {}     # name -> first elapsed (persistent)

    @classmethod
    def start(cls, name: str, measure_gpu: bool = True):
        with cls._lock:
            # Auto-reset: if results were already collected (by Report),
            # this is a new prompt execution — clear everything.
            if cls._collected:
                cls._timers.clear()
                # stop any orphaned pollers from a previous run
                for p in cls._pollers.values():
                    p.stop()
                cls._pollers.clear()
                cls._results.clear()
                cls._collected = False
            cls._timers[name] = time.perf_counter()
            if measure_gpu and torch.cuda.is_available():
                poller = _GPUPoller()
                poller.start()
                cls._pollers[name] = poller

    @classmethod
    def stop(cls, name: str, measure_gpu: bool = True):
        """Returns (elapsed_s, peak_used_bytes|None, total_bytes|None)
        or None if no matching start."""
        now = time.perf_counter()
        with cls._lock:
            start = cls._timers.pop(name, None)
            if start is None:
                return None
            elapsed = now - start

            peak_used = None
            total = None
            poller = cls._pollers.pop(name, None)
            if poller is not None:
                peak_used, total = poller.stop()

            cls._results.append((name, elapsed, peak_used, total))
            if name not in cls._cold_starts:
                cls._cold_starts[name] = elapsed
            return (elapsed, peak_used, total)

    @classmethod
    def get_results(cls) -> list[tuple]:
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
                "measure_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also measure GPU memory used between Start and Stop "
                               "(driver view — includes TensorRT engine memory)",
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

    def start_timer(self, passthrough, timer_name="timer_1", cuda_sync=True,
                    measure_gpu=True):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        _TimerStore.start(timer_name, measure_gpu=measure_gpu)
        gpu_note = ""
        if measure_gpu:
            used = _gpu_used_bytes()
            if used is not None:
                gpu_note = (f" (GPU in use {used / 1024**3:.2f} GiB; "
                            f"polling peak until Stop)")
        print(f"[qlip timer] '{timer_name}' started{gpu_note}")
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
                "measure_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also report PEAK total GPU memory in use between "
                               "Start and Stop (driver view — includes TensorRT "
                               "engine memory; sampled by a background poller so "
                               "it catches the mid-sampling peak)",
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

    def stop_timer(self, passthrough, timer_name="timer_1", cuda_sync=True,
                   measure_gpu=True):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        res = _TimerStore.stop(timer_name, measure_gpu=measure_gpu)
        if res is not None:
            elapsed, peak_used, total = res
            ms = elapsed * 1000
            text = f"{timer_name}: {elapsed:.3f} s ({ms:.1f} ms)"
            if peak_used is not None and total is not None:
                text += (f" | GPU peak {peak_used / 1024**3:.2f} / "
                         f"{total / 1024**3:.2f} GiB")
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

        for entry in results:
            # Tolerate both old (name, elapsed) and new
            # (name, elapsed, peak_used, total) shapes.
            name, elapsed = entry[0], entry[1]
            peak_used = entry[2] if len(entry) > 2 else None
            gpu_total = entry[3] if len(entry) > 3 else None
            ms = elapsed * 1000
            total += elapsed

            gpu_str = ""
            if peak_used is not None and gpu_total is not None:
                gpu_str = (f"  | GPU peak {peak_used / 1024**3:.2f} / "
                           f"{gpu_total / 1024**3:.2f} GiB")

            if track_cold_start:
                cold = _TimerStore.get_cold_start(name)
                if cold is not None:
                    cold_total += cold
                    has_cold = True
                    if abs(cold - elapsed) < 1e-6:
                        lines.append(f"  {name}: {elapsed:.3f} s ({ms:.1f} ms)  (cold start){gpu_str}")
                    else:
                        delta_pct = ((elapsed - cold) / cold) * 100
                        lines.append(
                            f"  {name}: {elapsed:.3f} s ({ms:.1f} ms)  "
                            f"(cold: {cold:.3f} s, {delta_pct:+.1f}%){gpu_str}"
                        )
                else:
                    lines.append(f"  {name}: {elapsed:.3f} s ({ms:.1f} ms){gpu_str}")
            else:
                lines.append(f"  {name}: {elapsed:.3f} s ({ms:.1f} ms){gpu_str}")

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
