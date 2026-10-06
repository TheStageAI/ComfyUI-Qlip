"""End-to-end check of ComfyUI-Qlip on Krea 2 Turbo through the ComfyUI HTTP API.

Builds each configuration as an API workflow (no UI export needed), restarts
ComfyUI per configuration (license binding and compiled state live in the
process), runs N prompts with a LoRA swap halfway, and records per-image
sampling time (QlipTimer), images, the cache report and any error.

    python tools/krea_check.py --comfy /workspace/ComfyUI --python /workspace/venv/bin/python \
        --configs eager engine engine_cache_prog compile_fp8 --n 8 --out /workspace/results/new

Side selection (A/B): --pythonpath prepends a qlip source tree (local build);
--nodes points custom_nodes/ComfyUI-Qlip at another checkout (symlink swap).

Models (ComfyUI/models): diffusion_models/krea2_turbo_bf16.safetensors,
text_encoders/qwen3vl_4b_bf16.safetensors, vae/qwen_image_vae.safetensors,
loras/krea2_retroanime.safetensors, loras/krea2_softwatercolor.safetensors
(all from HF Comfy-Org/Krea-2). Engines: HF TheStageAI/Elastic-Krea-2
models/H200/krea2-fp8_lora (downloaded by QlipEnginesLoader on first use).
"""
import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
import urllib.request
import uuid

PROMPTS = [
    "A cinematic high-fashion editorial portrait of a young woman in a structured black jacket inside a brutalist concrete gallery, soft window light, realistic skin texture",
    "Close-up beauty portrait of a woman with freckles, golden hour backlight, shallow depth of field, 85mm photo",
    "Full-body fashion shot of a model in a red silk dress on a rooftop at dusk, city lights bokeh, editorial",
    "Black and white studio portrait of a man in a tailored suit, dramatic rim light, high contrast",
    "A woman in a beige trench coat walking through a rainy Tokyo street at night, neon reflections, cinematic",
    "Portrait of an elderly fisherman with weathered skin, overcast light, documentary photography",
    "Editorial photo of a dancer mid-jump in a white studio, motion, crisp detail, high key lighting",
    "A young man in a denim jacket sitting on a vintage motorcycle in a desert, harsh noon sun, film grain",
]
LORAS = ["krea2_retroanime.safetensors", "krea2_softwatercolor.safetensors"]
HF_ENGINES = "TheStageAI/Elastic-Krea-2:models/H200/krea2-fp8_lora"
SIZE, STEPS, SEED = 1536, 8, 42026001


# ---------------------------------------------------------------------------
# workflow construction
# ---------------------------------------------------------------------------
def base_graph(size):
    """Loaders, text, latent, sampler, decode, timers. MODEL input of the
    sampler is wired by the caller (key "21" input "model")."""
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "krea2_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_4b_bf16.safetensors",
                "type": "krea2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        "10": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["2", 0]},
        },
        "11": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["10", 0]},
        },
        "20": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": size, "height": size, "batch_size": 1},
        },
        "24": {
            "class_type": "QlipTimerStart",
            "inputs": {
                "timer_name": "sampling_qlip",
                "cuda_sync": True,
                "measure_gpu": True,
                "passthrough": ["20", 0],
            },
        },
        "21": {
            "class_type": "KSampler",
            "inputs": {
                "seed": SEED,
                "steps": STEPS,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["24", 0],
            },
        },
        "25": {
            "class_type": "QlipTimerStop",
            "inputs": {
                "timer_name": "sampling_qlip",
                "cuda_sync": True,
                "measure_gpu": True,
                "passthrough": ["21", 0],
            },
        },
        "22": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["25", 0], "vae": ["3", 0]},
        },
        "23": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "krea_check", "images": ["22", 0]},
        },
    }


def chain(g, model_src, nodes):
    """Append MODEL->MODEL nodes after model_src; returns the last output."""
    src = model_src
    for nid, cls, inputs in nodes:
        g[nid] = {"class_type": cls, "inputs": dict(inputs, model=src)}
        src = [nid, 0]
    return src


CACHE = (
    "40",
    "QlipCache",
    {
        "enable": True,
        "threshold": 0.25,
        "mode": "step",
        "method": "easycache",
        "order": 2,
        "warmup_steps": 2,
        "max_consecutive_skips": 1,
        "fn_blocks": 1,
        "bn_blocks": 0,
    },
)
CACHE_BLOCK = (
    "40",
    "QlipCache",
    {
        "enable": True,
        "threshold": 0.08,
        "mode": "block",
        "method": "easycache",
        "order": 2,
        "warmup_steps": 2,
        "max_consecutive_skips": 2,
        "fn_blocks": 4,
        "bn_blocks": 0,
    },
)
PROG = (
    "41",
    "QlipProgressive",
    {
        "enable": True,
        "low_scale": 0.8,
        "switch_at": 0.3,
        "switch_mode": "sigma",
        "stab_threshold": 0.01,
    },
)
PROG_AUTO = (
    "41",
    "QlipProgressive",
    {
        "enable": True,
        "low_scale": 0.5,
        "switch_at": 0.5,
        "switch_mode": "auto",
        "stab_threshold": 0.08,
    },
)
SPARSE = (
    "42",
    "QlipAutoSparse",
    {
        "enable": True,
        "sparsity": 0.5,
        "selector": "diversity",
        "tau": 1.0,
        "correction": "none",
        "simthreshd1": 0.1,
        "smooth_k": True,
    },
)
# the krea2_new_release workflow: spectral auto ladder from 0.25 + step cache 0.3
PROG_RELEASE = (
    "41",
    "QlipProgressive",
    {
        "enable": True,
        "low_scale": 0.25,
        "switch_at": 0.3,
        "switch_mode": "auto",
        "stab_threshold": 0.01,
        "verify_sigma": 0.0,
        "carry_prev": True,
        "up_mode": "bilinear",
        "backbone_mode": "spectral",
        "speed_delta": 0.01,
        "speed_A": 0.024,
        "speed_beta": 2.55,
    },
)
CACHE_RELEASE = (
    "40",
    "QlipCache",
    {
        "enable": True,
        "threshold": 0.3,
        "mode": "step",
        "method": "easycache",
        "order": 2,
        "warmup_steps": 2,
        "max_consecutive_skips": 1,
        "fn_blocks": 1,
        "bn_blocks": 0,
    },
)
PRUNE = (
    "43",
    "QlipTokenPrune",
    {
        "enable": True,
        "keep_ratio": 0.75,
        "method": "l2sq",
        "compensation": "prev",
        "step_lo": 0.2,
        "step_hi": 0.8,
    },
)


def compile_node(quant, lora_mode=None, engines_dir=""):
    inp = {
        "enable": True,
        "quantize": quant,
        "backend": "default",
        "act_scales": "calibrate-first-run",
        "force_resident": True,
        "attention": "comfy",
        "weights_policy": "keep",
        "engines_dir": engines_dir,
    }
    if lora_mode:
        inp["lora_mode"] = lora_mode
    return ("44", "QlipCompile", inp)


def build(config, lora, size=SIZE, local_engines=""):
    """API workflow for one configuration; LoRA applied the way that path does it.
    engine* / release use the HF engines, local_* the engines in --local-engines."""
    g = base_graph(size)
    unet = ["1", 0]
    if config.startswith(("engine", "release", "local_")):
        g["31"] = {
            "class_type": "QlipLoraStack",
            "inputs": {"lora_path": f"__LORAS__/{lora}", "strength": 1.0},
        }
        local = config.startswith("local_")
        g["27"] = {
            "class_type": "QlipEnginesLoader",
            "inputs": {
                "model": unet,
                "engines_path": local_engines if local else "",
                "hf_repo": "" if local else HF_ENGINES,
                "cuda_graph": True,
                "shared_memory": "",
            },
        }
        # same wiring as the validated glam workflow: engines -> LoraSwitch -> Progressive -> Cache
        g["30"] = {
            "class_type": "QlipLoraSwitch",
            "inputs": {"model": ["27", 0], "enable": True, "lora_stack": ["31", 0]},
        }
        src = ["30", 0]
        extra = {
            "engine": [],
            "engine_cache": [CACHE],
            "engine_prog": [PROG],
            "engine_cache_prog": [PROG, CACHE],
            "release": [PROG_RELEASE, CACHE_RELEASE],
            "local_engine": [],
            "local_release": [PROG_RELEASE, CACHE_RELEASE],
        }[config]
    else:
        if config.startswith("compile") and "swap" in config:
            # PEFT side path: LoRA goes into QlipCompile, not ComfyUI's patcher
            g["31"] = {
                "class_type": "QlipLoraStack",
                "inputs": {"lora_path": f"__LORAS__/{lora}", "strength": 1.0},
            }
            src = unet
        else:
            g["5"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": unet, "lora_name": lora, "strength_model": 1.0},
            }
            src = ["5", 0]
        extra = {
            "eager": [],
            "eager_cache": [CACHE],
            "eager_cache_block": [CACHE_BLOCK],
            "eager_prog": [PROG],
            "eager_prog_auto": [PROG_AUTO],
            "eager_sparse": [SPARSE],
            "eager_prune": [PRUNE],
            "compile_bf16": [compile_node("none")],
            "compile_fp8": [compile_node("fp8")],
            "compile_fp8_swap": [compile_node("fp8", lora_mode="swap")],
            "compile_fp8_aot": [compile_node("fp8", engines_dir="__AOT__")],
            "compile_fp8_cache_prog": [compile_node("fp8"), PROG, CACHE],
            "compile_fp8_sparse": [SPARSE, compile_node("fp8")],
        }[config]
    out = chain(g, src, extra)
    if "44" in g and "31" in g:
        g["44"]["inputs"]["lora_stack"] = ["31", 0]
    g["21"]["inputs"]["model"] = out
    if "40" in g:
        g["26"] = {"class_type": "QlipCacheReport", "inputs": {"trigger": ["25", 0]}}
    return g


def fit_to_server(g, info):
    """Drop inputs the server's node does not have; fill missing required
    widget inputs with their defaults (HEAD vs new nodes differ)."""
    for nid, node in list(g.items()):
        spec = info.get(node["class_type"])
        if spec is None:
            raise RuntimeError(
                f"node {node['class_type']} is not registered on the server"
            )
        req = spec["input"].get("required", {})
        opt = spec["input"].get("optional", {})
        known = set(req) | set(opt)
        node["inputs"] = {k: v for k, v in node["inputs"].items() if k in known}
        for k, v in req.items():
            if k in node["inputs"]:
                continue
            cfg = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            if "default" in cfg:
                node["inputs"][k] = cfg["default"]
            elif isinstance(v[0], list) and v[0]:
                node["inputs"][k] = v[0][0]
    return g


# ---------------------------------------------------------------------------
# ComfyUI process + API
# ---------------------------------------------------------------------------
class Comfy:
    def __init__(self, a):
        self.a = a
        self.url = f"http://127.0.0.1:{a.port}"
        self.proc = None

    def start(self, log):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if self.a.pythonpath:
            env["PYTHONPATH"] = (
                self.a.pythonpath + os.pathsep + env.get("PYTHONPATH", "")
            )
        self.proc = subprocess.Popen(
            [
                self.a.python,
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                str(self.a.port),
                "--disable-auto-launch",
                "--output-directory",
                self.a.out_images,
            ],
            cwd=self.a.comfy,
            env=env,
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited, see {log}")
            try:
                urllib.request.urlopen(self.url + "/system_stats", timeout=3).read()
                return
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise RuntimeError("ComfyUI did not start in 600 s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def get(self, path):
        return json.load(urllib.request.urlopen(self.url + path, timeout=60))

    def run(self, g, timeout=3600):
        body = json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode()
        req = urllib.request.Request(
            self.url + "/prompt", body, {"Content-Type": "application/json"}
        )
        try:
            pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
        except urllib.error.HTTPError as e:
            return {"error": "validation: " + e.read().decode()[:2000]}
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self.get(f"/history/{pid}")
            if pid in h:
                return h[pid]
            time.sleep(0.5)
        return {"error": "timeout"}


TIMER_RE = re.compile(r"^\s*([\w.\-/]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s\b")


def parse(hist):
    st = hist.get("status", {})
    err = None
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                err = f"{m[1].get('exception_type')}: {m[1].get('exception_message')}"[
                    :600
                ]
    timers, texts, images = {}, [], []
    for out in hist.get("outputs", {}).values():
        for block in out.get("text", []) or []:
            if isinstance(block, str):
                texts.append(block)
                for line in block.splitlines():
                    m = TIMER_RE.match(line)
                    if m:
                        timers[m.group(1)] = float(m.group(2))
        for im in out.get("images", []) or []:
            images.append(os.path.join(im.get("subfolder", ""), im["filename"]))
    return {
        "error": hist.get("error") or err,
        "timers": timers,
        "texts": texts,
        "images": images,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="/workspace/ComfyUI")
    ap.add_argument("--python", default="/workspace/venv/bin/python")
    ap.add_argument(
        "--pythonpath", default="", help="qlip source tree to put first (local build)"
    )
    ap.add_argument(
        "--nodes",
        default="",
        help="ComfyUI-Qlip checkout to link as custom_nodes/ComfyUI-Qlip",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["eager", "engine", "engine_cache_prog", "compile_fp8"],
    )
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--local-engines", default="", help="engines dir for the local_* configs"
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.out_images = os.path.join(a.out, "images")
    os.makedirs(a.out_images, exist_ok=True)
    if a.nodes:
        link = os.path.join(a.comfy, "custom_nodes", "ComfyUI-Qlip")
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            raise SystemExit(f"{link} is a real directory: move it away to use --nodes")
        os.symlink(os.path.abspath(a.nodes), link)
    loras_dir = os.path.join(a.comfy, "models", "loras")
    results = {"args": vars(a), "configs": {}}
    for cfg in a.configs:
        c = Comfy(a)
        log = os.path.join(a.out, f"comfy_{cfg}.log")
        rec = {"runs": []}
        try:
            c.start(log)
            info = c.get("/object_info")
            rec["qlip_nodes"] = sorted(k for k in info if k.startswith("Qlip"))
            for i in range(a.n):
                lora = LORAS[0] if i < a.n // 2 else LORAS[1]
                g = build(cfg, lora, a.size, a.local_engines)
                g["10"]["inputs"]["text"] = PROMPTS[i % len(PROMPTS)]
                g["21"]["inputs"]["seed"] = SEED + i
                g["23"]["inputs"]["filename_prefix"] = f"{cfg}/{i:02d}"
                if "44" in g and g["44"]["inputs"].get("engines_dir") == "__AOT__":
                    g["44"]["inputs"]["engines_dir"] = os.path.join(a.out, "aot_pkgs")
                for n in g.values():
                    if n["class_type"] == "QlipLoraStack":
                        n["inputs"]["lora_path"] = n["inputs"]["lora_path"].replace(
                            "__LORAS__", loras_dir
                        )
                g = fit_to_server(g, info)
                t0 = time.time()
                r = parse(c.run(g))
                r["wall_s"] = round(time.time() - t0, 3)
                r["lora"] = lora
                rec["runs"].append(r)
                print(
                    f"[{cfg}] #{i} lora={lora[:18]} sampling={r['timers'].get('sampling_qlip')} wall={r['wall_s']} "
                    f"err={r['error']}",
                    flush=True,
                )
                if r["error"] and i == 0:
                    break
        except Exception as e:  # noqa: BLE001
            rec["fatal"] = repr(e)[:1000]
            print(f"[{cfg}] FATAL {e}", flush=True)
        finally:
            c.stop()
        ok = [
            r["timers"]["sampling_qlip"]
            for r in rec["runs"][1:]
            if not r["error"] and "sampling_qlip" in r["timers"]
        ]
        rec["steady_median_s"] = round(statistics.median(ok), 3) if ok else None
        rec["first_s"] = (
            rec["runs"][0]["timers"].get("sampling_qlip") if rec["runs"] else None
        )
        rec["first_wall_s"] = rec["runs"][0]["wall_s"] if rec["runs"] else None
        rec["errors"] = sum(1 for r in rec["runs"] if r["error"])
        results["configs"][cfg] = rec
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("\nconfig                     first_s  steady_median_s  errors")
    for cfg, rec in results["configs"].items():
        print(
            f"{cfg:26s} {str(rec.get('first_s')):>8} {str(rec.get('steady_median_s')):>16} {rec.get('errors')!s:>7}"
            + (f"  FATAL {rec['fatal'][:80]}" if rec.get("fatal") else "")
        )


if __name__ == "__main__":
    main()
