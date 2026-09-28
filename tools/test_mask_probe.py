# SPDX-License-Identifier: Apache-2.0
"""GPU probe: verify post-eviction causal masks + position separation end-to-end.

Review finding #3: the old probe only saved the first two explicit masks (both
pre-eviction). This version captures EVERY chunk x layer SDPA call and verifies:
  - chunk kwargs: physical cache_position = arange(phys_before, phys_before+b),
    original position_ids = arange(start, end)  (mask vs RoPE separation)
  - chunk 0: square shortcut (mask=None + is_causal) is allowed
  - chunks >=1: explicit mask row j == (slot <= phys_before + j), i.e. all
    survivors visible + intra-block causal; NO future token leak
    (old bug: first query of chunk 3 saw 768 keys instead of 513)

Config: L=1024, B=256, K=512 -> phys_before per chunk = [0, 256, 512, 512];
first eviction happens during chunk 2, so chunks 2 AND 3 are post-eviction.
"""

import hashlib
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", os.path.join(HERE, "cache", "hf"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False  # repeat_kv patch (stage-1 harness)

import presses.chunked_online_press as P  # noqa: E402

FUSED = os.environ.get("CHUNKED_FUSED", "0") == "1"  # run the whole suite through the NVRTC fused kernel


def ChunkedPress(**kw):
    p = P.ChunkedOnlinePress(**kw)
    if FUSED:
        p.fused_scoring = True
    return p


FAILURES = []


def check(name, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


SRC = Path(P.__file__)
print(f"press source : {SRC}")
print(f"sha256       : {hashlib.sha256(SRC.read_bytes()).hexdigest()}")
print(f"torch        : {torch.__version__} | transformers: {transformers.__version__}")
assert SRC.resolve().is_relative_to(Path(HERE).resolve()), f"press module not from this tree: {SRC}"

from transformers import AutoModelForCausalLM  # noqa: E402

MODEL = os.environ.get("KVPROVE_MODEL_PATH",
          r"D:\mxy\llm\kvpress_project\models\Qwen3-8B")  # configurable; override with env var
L, B, RATIO = 1024, 256, 0.5
K = 512  # compute_n_kept(1024, 0.5)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda")
model.eval()
NL = model.config.num_hidden_layers
print(f"layers={NL}")

# ---- instruments ----
chunk_calls = []   # per inner (chunk) forward: cache_position / position_ids actually used
sdpa_calls = []    # per attention call: (q_len, kv_len, is_causal, mask_bool or None)

_orig_sdpa = F.scaled_dot_product_attention


def probe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    if attn_mask is None:
        vis = None
    elif attn_mask.dtype == torch.bool:
        vis = attn_mask.detach().to("cpu", torch.bool)
    else:  # additive float mask: > -1e3 means visible
        vis = attn_mask.detach().to("cpu", torch.float32) > -1e3
    sdpa_calls.append((query.shape[-2], key.shape[-2], bool(is_causal), vis))
    return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)


F.scaled_dot_product_attention = probe_sdpa

real_forward = model.model.forward


def probe_forward(*args, **kwargs):
    cp = kwargs.get("cache_position")
    pid = kwargs.get("position_ids")
    chunk_calls.append({
        "cache_position": None if cp is None else cp.detach().cpu(),
        "position_ids": None if pid is None else pid.detach().cpu(),
    })
    return real_forward(*args, **kwargs)


model.model.forward = probe_forward  # press wraps THIS -> probe sees chunk-level kwargs

press = ChunkedPress(compression_ratio=RATIO, block_length=B, mode="replace")
torch.manual_seed(0)
ids = torch.randint(1000, 20000, (1, L), device="cuda")

with torch.no_grad():
    with press(model):
        model.model(input_ids=ids, past_key_values=None)

F.scaled_dot_product_attention = _orig_sdpa
model.model.forward = real_forward

# ---- expectations ----
n_chunks = L // B
phys_before = [min(c * B, K) for c in range(n_chunks)]
print(f"expected physical length before each chunk: {phys_before}")

ok_n = len(chunk_calls) == n_chunks
check("M0.chunk_call_count", ok_n, f"got {len(chunk_calls)} inner forwards, want {n_chunks}")

if ok_n:
    for c, rec in enumerate(chunk_calls):
        cp, pid = rec["cache_position"], rec["position_ids"]
        want_cp = torch.arange(phys_before[c], phys_before[c] + B)
        want_pid = torch.arange(c * B, (c + 1) * B).unsqueeze(0)
        check(f"M1.chunk{c}.cache_position_physical",
              cp is not None and torch.equal(cp, want_cp),
              f"got={None if cp is None else cp[:3].tolist()}... want={want_cp[:3].tolist()}...")
        check(f"M1.chunk{c}.position_ids_original",
              pid is not None and torch.equal(pid, want_pid),
              f"got={None if pid is None else pid[0][:3].tolist()}... want={want_pid[0][:3].tolist()}...")

n_attn = len(sdpa_calls)
check("M2.attention_call_count", n_attn == n_chunks * NL, f"got {n_attn}, want {n_chunks * NL}")

if n_attn == n_chunks * NL:
    bad_len, bad_mask = [], []
    for c in range(n_chunks):
        for l in range(NL):
            q_len, kv_len, is_causal, mask = sdpa_calls[c * NL + l]
            exp_kv = phys_before[c] + B
            if (q_len, kv_len) != (B, exp_kv):
                bad_len.append((c, l, q_len, kv_len))
                continue
            if c == 0:
                if not (mask is None and is_causal):
                    bad_mask.append((c, l, "chunk0-not-square-shortcut"))
                continue
            if mask is None or is_causal:
                bad_mask.append((c, l, "missing-explicit-mask"))
                continue
            m2 = mask[0, 0]  # (q, kv) True=visible
            want = torch.arange(exp_kv)[None, :] <= (phys_before[c] + torch.arange(B))[:, None]
            if not torch.equal(m2, want):
                leak = int((m2 & ~want).sum())
                missing = int((~m2 & want).sum())
                bad_mask.append((c, l, f"leak={leak} missing={missing} row0={int(m2[0].sum())}"))
    check("M3.kv_shapes_per_chunk", not bad_len, f"bad={bad_len[:4]}")
    check("M4.post_eviction_mask_pattern", not bad_mask,
          (f"bad={bad_mask[:4]}" if bad_mask else
           f"all {n_chunks * NL} verified; chunk3 row0 sees {phys_before[3] + 1} keys (was 768 before fix)"))

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("MASK/POSITION PROBE PASSED")
