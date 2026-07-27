#!/usr/bin/env python3
"""
GLM-5.2 (glm_moe_dsa) -> JANGTQ converter.  Profiles: 2L (default) and K.

Mirrors convert_glm51_jangtq_2l.py (same proven on-disk format), specialized for GLM-5.2.

Shared control-plane policy (both profiles):
  * MoE router gate + e_score_correction_bias, all norms, all biases -> FP16 passthrough.
  * First first_k_dense_replace dense MLP layers -> control-plane precision (not routed).

Profile 2L (uniform):
  * Routed experts            -> 2-bit MXTQ (TurboQuant: random-sign Hadamard + Lloyd-Max codebook).
  * Attention (MLA + DSA indexer), shared_experts, embed_tokens, lm_head -> FP16 passthrough.

Profile K (max-quality, house JANGTQ_K asymmetry):
  * Routed experts            -> down_proj 4-bit, gate_proj/up_proj 2-bit (all MXTQ).
                                 (matches documented MiniMax / Qwen3.5 / Hunyuan3 / Kimi K budget;
                                  gate==up is required by the fused gate_up kernel; widths in {2,4}
                                  only, since 3-bit over-rounds on GLM's 6144/2048 dims.)
  * Non-routed                -> FP16 by default (GLM-2L MLA-drift fix); --k-attn-8bit switches
                                 attention/shared/embed/lm_head to 8-bit affine (literal house K, smaller).
  * MTP head                  -> always dropped (speculative-decode only; the MLX/JANGTQ decode
                                 path ignores it).

GLM-5.2 specifics: IndexShare "shared" attention layers carry NO indexer weights on disk -- index
*reuse* is a forward-pass concern (full layers compute indices, shared layers reuse them), not a
convert-time one. The indexer weights that DO exist (full layers) are kept at the control-plane
precision. MTP lives at layer index == num_hidden_layers (eh_proj/enorm/hnorm submodules).

Runs as a plain script regardless of editable-install state (stable core modules only).
AWQ optional: without --awq-scales it still emits a valid codebook+Hadamard bundle (RTN fallback).

    python3 convert_glm52_jangtq.py --dry-run /Volumes/TB5/llm/GLM-5.2
    python3 convert_glm52_jangtq.py --dry-run --profile K /Volumes/TB5/llm/GLM-5.2
    python3 convert_glm52_jangtq.py --profile K /Volumes/TB5/llm/GLM-5.2 /Volumes/TB5/llm/GLM-5.2-JANGTQ_K
"""
import sys, os, json, gc, shutil, re, struct, argparse, collections
from pathlib import Path
import numpy as np

SEED = 42
FP8_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E4M3FN"}
_EXPERT_TO_SWITCH = re.compile(r"\.experts\.\d+\.")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_MTP_NAME_RE = re.compile(r"(^|\.)(mtp|nextn|eh_proj|enorm|hnorm|shared_head)(\.|_|$)", re.I)

# House JANGTQ_K routed budget: gate==up (fused-kernel requirement), down higher.
# Widths restricted to {2,4}: 3-bit -> vals_per_u32=10 over-rounds on GLM in_feat 6144/2048.
ROUTED_K_BITS = {"gate_proj": 2, "up_proj": 2, "down_proj": 4}


def awq_key_for(tensor_name: str) -> str:
    base = tensor_name.replace(".weight", "")
    return _EXPERT_TO_SWITCH.sub(".switch_mlp.", base)


def _routed_proj(name: str):
    low = name.lower()
    if "down_proj" in low: return "down_proj"
    if "gate_proj" in low: return "gate_proj"
    if "up_proj" in low:   return "up_proj"
    return None


def get_bits_and_method(tensor_name: str, profile: str = "2L", k_attn_8bit: bool = False):
    """(bits, method) in {'mxtq','mxtq_awq','affine','passthrough'}."""
    name = tensor_name.lower()
    # control plane shared by all profiles
    if "norm" in name or tensor_name.endswith(".bias"):
        return (16, "passthrough")
    if ".gate." in name and "gate_proj" not in name:        # MoE router gate + e_score_correction_bias
        return (16, "passthrough")
    # routed experts
    if ("experts" in name and "shared_expert" not in name) or "switch_mlp" in name:
        if profile == "K":
            return (ROUTED_K_BITS.get(_routed_proj(name), 2), "mxtq_awq")
        if profile == "4":
            return (4, "mxtq_awq")                          # JANGTQ4: uniform 4-bit
        return (2, "mxtq_awq")                              # 2L: uniform 2-bit
    # non-routed every-token path
    nonrouted = (8, "affine") if (profile == "K" and k_attn_8bit) else (16, "passthrough")
    if "embed_tokens" in name or "lm_head" in name:
        return nonrouted
    if "self_attn" in name:                                 # MLA + DSA indexer
        return nonrouted
    if "shared_expert" in name:
        return nonrouted
    return nonrouted                                        # first-k dense MLP + anything else


def is_mtp_tensor(tensor_name: str, n_layers: int) -> bool:
    if _MTP_NAME_RE.search(tensor_name):
        return True
    m = _LAYER_RE.search(tensor_name)
    return bool(m and int(m.group(1)) >= n_layers)


def tier_label(name: str, n_layers: int, profile: str = "2L", k_attn_8bit: bool = False) -> str:
    low = name.lower()
    if is_mtp_tensor(name, n_layers):
        return "mtp(dropped in K)"
    if "experts" in low and "shared_expert" not in low:
        if profile == "K":
            p = _routed_proj(name)
            return f"routed_{p or '?'}({ROUTED_K_BITS.get(p,2)}b)"
        if profile == "4":
            return "routed_expert(4b)"
        return "routed_expert(2b)"
    nb = "8b-affine" if (profile == "K" and k_attn_8bit) else "fp16"
    if "shared_expert" in low: return f"shared_expert({nb})"
    if "indexer" in low:       return f"dsa_indexer({nb})"
    if "self_attn" in low:     return f"attention({nb})"
    if ".gate." in low and "gate_proj" not in low: return "router/gate(fp16)"
    if "embed_tokens" in low or "lm_head" in low:  return f"embed/head({nb})"
    if "norm" in low or name.endswith(".bias"):     return "norm/bias(fp16)"
    m = _LAYER_RE.search(name)
    if m and int(m.group(1)) < n_layers and ".mlp." in low and "expert" not in low:
        return f"dense_mlp({nb})"
    return "OTHER(fp16?)"


class _Headers:
    def __init__(self): self.cache = {}
    def get(self, path):
        p = str(path)
        if p not in self.cache:
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.cache[p] = json.loads(fh.read(n))
        return self.cache[p]
    def dtype_of(self, path, name):
        e = self.get(path).get(name)
        return e.get("dtype") if e else None


def _norm_pattern(k):
    k = _LAYER_RE.sub(".layers.{L}.", k)
    return re.sub(r"\.experts\.\d+\.", ".experts.{E}.", k)


def scan_source(SRC, headers):
    all_tensors, seen, dt_hist = [], set(), collections.Counter()
    shards = sorted(SRC.glob("model-*.safetensors"))
    for sf in shards:
        for k, meta in headers.get(sf).items():
            if k == "__metadata__" or k.endswith("_scale_inv"):
                continue
            dt_hist[meta.get("dtype", "?")] += 1
            all_tensors.append((k, list(meta.get("shape", [])), sf))
            m = _LAYER_RE.search(k)
            if m: seen.add(int(m.group(1)))
    return all_tensors, seen, dt_hist, shards


def preflight(SRC, n_layers, seen_layers, shards):
    issues = []
    idx = SRC / "model.safetensors.index.json"
    if idx.exists():
        wm = json.load(open(idx)).get("weight_map", {})
        missing = sorted(set(wm.values()) - {p.name for p in shards})
        if missing:
            issues.append(f"{len(missing)} shard(s) in index.json missing on disk, e.g. {missing[:3]}")
    else:
        issues.append("no model.safetensors.index.json present (download may be incomplete)")
    ml = sorted(set(range(n_layers)) - seen_layers)
    if ml:
        issues.append(f"{len(ml)}/{n_layers} transformer layers absent, e.g. {ml[:8]}")
    return issues


def run_dry(SRC, config, profile="2L", k_attn_8bit=False):
    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]; n_experts = tc.get("n_routed_experts", 256)
    first_dense = tc.get("first_k_dense_replace", 0); n_mtp = tc.get("num_nextn_predict_layers", 0)
    headers = _Headers(); all_t, seen, dt_hist, shards = scan_source(SRC, headers)

    print("=" * 64)
    print(f"  GLM-5.2 JANGTQ_{profile}  --  DRY RUN (no writes)")
    print("=" * 64)
    print(f"  source: {SRC}")
    print(f"  shards on disk: {len(shards)}   tensors scanned: {len(all_t)}")
    print(f"  source dtypes: {dict(dt_hist)}")
    if profile == "K":
        nb = "8-bit affine" if k_attn_8bit else "FP16"
        print(f"  profile K: routed gate=2/up=2/down=4 MXTQ, non-routed={nb}, MTP dropped")
    elif profile == "4":
        print("  profile 4 (JANGTQ4): routed 4-bit MXTQ, non-routed FP16, MTP dropped by default")
    else:
        print("  profile 2L: routed 2-bit MXTQ, non-routed FP16, MTP dropped by default")
    print(f"  config: {n_layers} layers ({first_dense} dense), {n_experts} experts, num_nextn_predict_layers={n_mtp}")
    print(f"  layer coverage: {len(seen)}/{n_layers} present"
          + ("" if len(seen) == n_layers else f"  (absent e.g. {sorted(set(range(n_layers))-seen)[:8]})"))

    issues = preflight(SRC, n_layers, seen, shards)
    print("\n  preflight:")
    print("    [OK] index + all layers present" if not issues else "")
    for s in issues: print(f"    [INCOMPLETE] {s}")

    methods, tiers, others, pack_bad = collections.Counter(), collections.Counter(), set(), []
    for name, shape, _ in all_t:
        bits, method = get_bits_and_method(name, profile, k_attn_8bit)
        methods[method] += 1
        lab = tier_label(name, n_layers, profile, k_attn_8bit)
        tiers[lab] += 1
        if lab == "OTHER(fp16?)": others.add(_norm_pattern(name))
        if method in ("mxtq", "mxtq_awq") and len(shape) == 2:
            vpu = 32 // bits
            if shape[1] % vpu != 0:
                pack_bad.append((name, shape, bits))

    print("\n  method totals:")
    for k in ("passthrough", "mxtq_awq", "mxtq", "affine"):
        if methods.get(k): print(f"    {k:12} {methods[k]}")
    print("\n  tier breakdown:")
    for k, v in sorted(tiers.items(), key=lambda kv: -kv[1]): print(f"    {k:24} {v}")

    moe_layers = sorted(L for L in seen if L >= first_dense); chk = None
    for L in moe_layers:
        eids = {int(m.group(1)) for name, _, _ in all_t
                for m in [re.search(rf"\.layers\.{L}\.mlp\.experts\.(\d+)\.", name)] if m}
        if eids: chk = (L, eids); break
    print("\n  expert contiguity:")
    if chk:
        L, eids = chk; miss = sorted(set(range(n_experts)) - eids)
        print(f"    [OK] layer {L}: all {n_experts} experts present" if not miss
              else f"    [INFO] layer {L}: {len(eids)}/{n_experts} experts on disk (e.g. missing {miss[:6]}) -- expected until DL completes")
    else:
        print("    [INFO] no MoE layer fully present yet")

    print("\n  routed-expert pack divisibility (in_feat % (32//bits) == 0):")
    print("    [OK] all routed-expert weights pack cleanly" if not pack_bad
          else f"    [WARN] {len(pack_bad)} would trip tq assert, e.g. {pack_bad[:3]}")

    mtp = sorted({_norm_pattern(n) for n, _, _ in all_t if is_mtp_tensor(n, n_layers)})
    print("\n  MTP tensors detected:", "none in present shards (tail shards)" if not mtp else f"{len(mtp)} patterns (will be DROPPED)")
    for p in mtp[:20]: print("     ", p)

    if others:
        print("\n  [REVIEW] tensors hitting the generic fp16 fallback (unexpected names):")
        for p in sorted(others)[:40]: print("     ", p)
    else:
        print("\n  [OK] no unclassified tensors -- every name maps to a known tier")
    print("\n  verdict:",
          "classifier maps cleanly; finish the download, then run the real conversion."
          if not others and not pack_bad else "review flagged items above before a real run.")
    print("=" * 64, flush=True)


def _setup_capabilities(jang_config, config, out_path):
    try:
        from jang_tools.capabilities import build_capabilities
        caps = build_capabilities(jang_config, config, out_path)
        if caps:
            jang_config["capabilities"] = caps
            print(f"  capabilities: family={caps.get('family')} modality={caps.get('modality')}", flush=True)
    except Exception as e:
        print(f"  [capabilities] skipped: {e}", flush=True)


def _load_tensor(headers, sf_path, name, shape, L):
    from safetensors import safe_open
    dt = headers.dtype_of(sf_path, name)
    try:
        if dt in FP8_DTYPES:
            with safe_open(str(sf_path), framework="numpy") as f:
                try: scale = f.get_tensor(name + "_scale_inv")
                except Exception: scale = None
            t = L["load_fp8_tensor"](sf_path, name, shape, scale)
        elif dt == "BF16":
            t = L["_load_bf16_tensor"](sf_path, name, shape)
        else:
            with safe_open(str(sf_path), framework="numpy") as f:
                t = f.get_tensor(name)
        t = np.asarray(t)
        return t.astype(np.float32) if t.dtype != np.float32 else t
    except Exception:
        with safe_open(str(sf_path), framework="numpy") as f:
            try: scale = f.get_tensor(name + "_scale_inv")
            except Exception: scale = None
            try:
                t = L["load_fp8_tensor"](sf_path, name, shape, scale)
            except Exception:
                try:
                    t = f.get_tensor(name)
                    if not isinstance(t, np.ndarray): t = np.array(t)
                except Exception:
                    t = L["_load_bf16_tensor"](sf_path, name, shape)
        t = np.asarray(t)
        return t.astype(np.float32) if t.dtype != np.float32 else t


def run_convert(SRC, OUT, config, awq_file, drop_mtp, max_shard, allow_incomplete,
                profile="2L", k_attn_8bit=False):
    import mlx.core as mx
    from jang_tools.fp8 import load_fp8_tensor
    from jang_tools.calibrate import _load_bf16_tensor
    from jang_tools.turboquant.linear import tq_quantize_weight
    from safetensors.numpy import save_file, load_file
    from tqdm import tqdm
    L = {"load_fp8_tensor": load_fp8_tensor, "_load_bf16_tensor": _load_bf16_tensor}

    if profile == "K":
        assert ROUTED_K_BITS["gate_proj"] == ROUTED_K_BITS["up_proj"], "K requires gate==up bits (fused kernel)"
        assert set(ROUTED_K_BITS.values()) <= {2, 4}, "K routed bits must be in {2,4} (3-bit over-rounds on GLM dims)"

    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]; first_dense = tc.get("first_k_dense_replace", 0)
    n_experts = tc.get("n_routed_experts", 256); n_mtp = tc.get("num_nextn_predict_layers", 0)

    headers = _Headers(); all_t, seen, dt_hist, shards = scan_source(SRC, headers)
    issues = preflight(SRC, n_layers, seen, shards)
    if issues and not allow_incomplete:
        print("REFUSING: source looks incomplete:")
        for s in issues: print("   -", s)
        print("Finish the download, or pass --allow-incomplete to override.")
        raise SystemExit(2)

    awq_scales = {}
    if awq_file:
        ap = Path(awq_file)
        if ap.exists():
            awq_scales = {k: v.astype(np.float32) for k, v in load_file(str(ap)).items()}
            print(f"  loaded AWQ scales for {len(awq_scales)} modules", flush=True)
        else:
            print(f"  WARNING: AWQ scales {ap} not found -> RTN fallback", flush=True)
    else:
        print("  no --awq-scales -> RTN fallback (still codebook+Hadamard)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    nb = ("8-bit affine" if k_attn_8bit else "FP16") if profile == "K" else "FP16"
    routed_desc = ("gate=2/up=2/down=4 MXTQ" if profile == "K"
                   else "4-bit MXTQ" if profile == "4" else "2-bit MXTQ")
    print("=" * 64)
    print(f"  GLM-5.2 -> JANGTQ_{profile}")
    print(f"  src={SRC}\n  out={OUT}")
    print(f"  {n_layers} layers ({first_dense} dense), {n_experts} experts, src dtypes={dict(dt_hist)}")
    print(f"  routed={routed_desc}  non-routed={nb}  MTP={'drop' if drop_mtp else f'keep({n_mtp})'}")
    print("=" * 64, flush=True)

    if drop_mtp:
        all_t = [(n, s, p) for (n, s, p) in all_t if not is_mtp_tensor(n, n_layers)]
    mtp_kept = sum(1 for n, _, _ in all_t if is_mtp_tensor(n, n_layers))

    shard_idx = 0; shard_tensors = {}; shard_bytes = 0; shard_map = {}
    totals = collections.Counter()

    def flush_shard():
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors: return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(OUT / fname))
        for k in shard_tensors: shard_map[k] = fname
        print(f"    shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.1f} GB", flush=True)
        shard_tensors = {}; shard_bytes = 0  # noqa: F841

    def add_tensor(name, arr):
        nonlocal shard_tensors, shard_bytes
        shard_tensors[name] = arr; shard_bytes += arr.nbytes
        if shard_bytes >= max_shard: flush_shard()

    awq_written = set()
    for name, shape, sf in tqdm(all_t, desc="  processing"):
        bits, method = get_bits_and_method(name, profile, k_attn_8bit)
        tensor = _load_tensor(headers, sf, name, shape, L)

        if method == "passthrough":
            add_tensor(name, tensor.astype(np.float16)); totals["passthrough"] += 1

        elif method == "affine":
            gs = 64
            if tensor.shape[-1] % gs != 0:
                add_tensor(name, tensor.astype(np.float16)); totals["passthrough_fallback"] += 1
            else:
                qw, qs, qb = mx.quantize(mx.array(tensor.astype(np.float16)), group_size=gs, bits=bits)
                base = name[:-7] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.array(qw))
                add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
                add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
                totals["affine"] += 1

        elif method in ("mxtq", "mxtq_awq"):
            scale = awq_scales.get(awq_key_for(name)) if method == "mxtq_awq" else None
            if scale is not None:
                if scale.shape[0] != tensor.shape[1]:
                    raise RuntimeError(f"AWQ scale shape mismatch {name}: in={tensor.shape[1]} scale={scale.shape[0]}")
                tensor = tensor * scale[None, :]; totals["awq_applied"] += 1
            result = tq_quantize_weight(tensor, bits=bits, seed=SEED)
            base = name[:-7] if name.endswith(".weight") else name
            add_tensor(f"{base}.tq_packed", result["packed"])
            add_tensor(f"{base}.tq_norms", result["norms"])
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.uint8))
            if scale is not None:
                ak = awq_key_for(name)
                if ak not in awq_written:
                    add_tensor(f"{ak}.awq_scale", scale.astype(np.float16)); awq_written.add(ak)
            totals[method] += 1
        del tensor
        if sum(totals[m] for m in ("mxtq", "mxtq_awq")) % 100 == 0:
            gc.collect()
    flush_shard()

    print(f"\n  writing index ({shard_idx} shards)...", flush=True)
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        if old.exists():
            old.rename(OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors")
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": shard_map}, f, indent=2)

    if profile == "K":
        routed_meta = {"gate_proj": ROUTED_K_BITS["gate_proj"], "up_proj": ROUTED_K_BITS["up_proj"],
                       "down_proj": ROUTED_K_BITS["down_proj"]}
        nb_bit = 8 if k_attn_8bit else 16; profile_name = "JANGTQ_K"
    elif profile == "4":
        routed_meta = 4; nb_bit = 16; profile_name = "JANGTQ4"
    else:
        routed_meta = 2; nb_bit = 16; profile_name = "JANGTQ_2L"

    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": 64, "bits": 8}
    config["weight_format"] = "mxtq"
    config["routed_expert_bits"] = routed_meta            # top-level: loaders/audits read this
    jang_config = {
        "weight_format": "mxtq", "profile": profile_name, "variant": "GLM-5.2",
        "model_family": "glm_moe_dsa", "index_share": True, "mxtq_seed": SEED,
        "mxtq_bits": {"attention": nb_bit, "shared_expert": nb_bit, "routed_expert": routed_meta,
                      "embed_tokens": nb_bit, "lm_head": nb_bit},
        "routed_expert_bits": routed_meta,
        "method": ("passthrough+affine+mxtq_awq" if (profile == "K" and k_attn_8bit) else "passthrough+mxtq_awq"),
        "awq_enabled": bool(awq_scales),
    }
    if not drop_mtp and (mtp_kept > 0 or n_mtp > 0):
        jang_config["mtp"] = {"kept": mtp_kept > 0, "enabled": mtp_kept > 0,
                              "num_layers": n_mtp, "tensor_count": mtp_kept}
        jang_config["bundle_has_mtp"] = mtp_kept > 0
    _setup_capabilities(jang_config, config, OUT)
    config["jang"] = jang_config
    with open(OUT / "config.json", "w") as f: json.dump(config, f, indent=2)
    with open(OUT / "jang_config.json", "w") as f: json.dump(jang_config, f, indent=2)

    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "chat_template.jinja", "chat_template.json",
               "merges.txt", "vocab.json", "preprocessor_config.json", "configuration.json"]:
        s = SRC / fn
        if s.exists(): shutil.copy2(str(s), str(OUT / fn))
    for s in SRC.glob("*.py"): shutil.copy2(str(s), str(OUT / s.name))

    tcf = OUT / "tokenizer_config.json"
    if tcf.exists():
        try:
            d = json.load(open(tcf))
            if d.get("tokenizer_class") == "TokenizersBackend":
                d["tokenizer_class"] = "ChatGLMTokenizer"
                json.dump(d, open(tcf, "w"), indent=2)
                print("  [osaurus-fix] tokenizer_class -> ChatGLMTokenizer", flush=True)
        except Exception as e:
            print(f"  [osaurus-fix] skipped: {e}", flush=True)

    print("\n  building jangtq_runtime.safetensors sidecar...", flush=True)
    try:
        from jang_tools.build_jangtq_sidecar import main as _sidecar
        saved = sys.argv; sys.argv = ["build_jangtq_sidecar", str(OUT)]
        try: _sidecar()
        finally: sys.argv = saved
    except (Exception, SystemExit) as e:
        print(f"  [sidecar] FAILED: {e}", flush=True)

    print(f"\n{'='*64}\n  DONE -- GLM-5.2 {profile_name} -> {OUT}")
    for k, v in totals.items(): print(f"    {k}: {v}")
    du = sum(f.stat().st_size for f in OUT.glob('*') if f.is_file())
    print(f"  total: {du/1e9:.1f} GB\n{'='*64}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GLM-5.2 glm_moe_dsa -> JANGTQ (2L | K)")
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--profile", default="2L", help="2L | K | 4  (aliases: JANGTQ_2L, JANGTQ_K/2K, JANGTQ4/tq4)")
    ap.add_argument("--k-attn-8bit", action="store_true",
                    help="K only: attention/shared/embed/lm_head at 8-bit affine (literal house K, smaller) instead of FP16")
    ap.add_argument("--dry-run", action="store_true", help="classify + preflight only, no writes")
    ap.add_argument("--awq-scales", default=None, help="optional AWQ scales .safetensors")
    ap.add_argument("--keep-mtp", action="store_true",
                    help="keep MTP/nextn tensors (2L only; ignored for K. Only useful for a runtime doing MTP "
                         "speculative decode -- the MLX/JANGTQ decode path ignores them, so default is to drop)")
    ap.add_argument("--allow-incomplete", action="store_true", help="convert even if shards/layers missing")
    ap.add_argument("--max-shard-gb", type=float, default=1.0)
    a = ap.parse_args(argv)

    prof = {"2L": "2L", "JANGTQ_2L": "2L", "JANGTQ2L": "2L",
            "K": "K", "2K": "K", "JANGTQ_K": "K", "JANGTQK": "K",
            "4": "4", "JANGTQ4": "4", "TQ4": "4", "4L": "4"}.get(a.profile.upper().strip())
    if prof is None:
        raise SystemExit(f"unknown --profile {a.profile!r}; use 2L or K")
    if a.k_attn_8bit and prof != "K":
        raise SystemExit("--k-attn-8bit only applies to --profile K")
    if prof == "K" and a.keep_mtp:
        print("note: JANGTQ_K drops the MTP head regardless of --keep-mtp (speculative-only; MLX decode ignores it)")
    drop_mtp = True if prof == "K" else (not a.keep_mtp)

    SRC = Path(a.src)
    with open(SRC / "config.json") as f: config = json.load(f)
    if a.dry_run:
        run_dry(SRC, config, prof, a.k_attn_8bit); return
    if not a.out:
        raise SystemExit("OUT directory required for a real run (or pass --dry-run).")
    run_convert(SRC, Path(a.out), config, a.awq_scales, drop_mtp,
                int(a.max_shard_gb * 1_000_000_000), a.allow_incomplete, prof, a.k_attn_8bit)


if __name__ == "__main__":
    main()
