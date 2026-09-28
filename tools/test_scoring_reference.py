# SPDX-License-Identifier: Apache-2.0
"""Independent reference test for the scoring rule, WITH eviction active (r=0.5).

Review gap fixed: the v1 reference test ran ratio=0 and compared only each
chunk's new columns. This version runs ratio=0.5 (evictions at chunks 2 and 3),
and compares the FULL score vector (survivor columns + current-block columns)
against the model's own eager attention weights for every (chunk, layer).

Method: mode='replace' makes the stored state scores equal the current chunk's
causal-corrected attention mass, so snapshotting press._state per (chunk, layer)
gives the exact vector to compare with the eager reference.
"""

import hashlib
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", os.path.join(HERE, "cache", "hf"))

import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False

import presses.chunked_online_press as P  # noqa: E402

FUSED = os.environ.get("CHUNKED_FUSED", "0") == "1"  # run the whole suite through the NVRTC fused kernel


def ChunkedPress(**kw):
    p = P.ChunkedOnlinePress(**kw)
    if FUSED:
        p.fused_scoring = True
    return p


SRC = Path(P.__file__)
print(f"press source : {SRC}")
print(f"sha256       : {hashlib.sha256(SRC.read_bytes()).hexdigest()}")
print(f"torch {torch.__version__} | transformers {transformers.__version__}")
assert SRC.resolve().is_relative_to(Path(HERE).resolve())

from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

MODEL = os.environ.get("KVPROVE_MODEL_PATH",
          r"D:\mxy\llm\kvpress_project\models\Qwen3-8B")  # configurable; override with env var
L, B, RATIO = 1024, 256, 0.5
K = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="eager"
).to("cuda")
model.eval()
NL = len(model.model.layers)
Hq = model.config.num_attention_heads
Hkv = model.config.num_key_value_heads
G = Hq // Hkv

# ---- capture structures ----
attn_cap = {}    # (chunk, layer) -> attn weights (Hq, b, kv) float32 cpu
state_cap = {}   # (chunk, layer) -> post-chunk press scores (Hkv, K) float32 cpu
sel_cap = {}     # (chunk, layer) -> PRE-eviction scores from _select args (Hkv, cand)
_cur_layer = [0]


def make_attn_hook(li):
    def hook(module, args, kwargs, output):
        c = press._chunk_idx
        att = None
        if hasattr(output, "attentions"):
            att = output.attentions
        elif isinstance(output, tuple):
            for o in output:
                # attention tensor: (bsz, Hq, q_len=B, kv_len) — kv may differ from q
                if torch.is_tensor(o) and o.dim() == 4 and o.shape[0] == 1 \
                        and o.shape[1] == Hq and o.shape[2] == B:
                    att = o
                    break
        if att is not None:
            attn_cap[(c, li)] = att.detach().float().cpu()
    return hook


def make_state_hook(li):
    def hook(module, args, kwargs, output):
        c = press._chunk_idx
        st = press._state.get(li)
        if st is not None:
            state_cap[(c, li)] = st["scores"].detach().float().cpu()
    return hook


def make_pre_hook(li):
    def pre_hook(module, args, kwargs):
        _cur_layer[0] = li
    return pre_hook


# spy on _select to capture the PRE-eviction score vector (full candidate columns)
_orig_select = P.ChunkedOnlinePress.__dict__["_select"]


def _spy_select(scores, cand_idx, K_, n_sink, W=0):
    sel_cap[(press._chunk_idx, _cur_layer[0])] = scores.detach().float().cpu()
    return _orig_select(scores, cand_idx, K_, n_sink, W)


P.ChunkedOnlinePress._select = staticmethod(_spy_select)

hooks = []
for li, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li), with_kwargs=True))
    hooks.append(layer.self_attn.register_forward_pre_hook(make_pre_hook(li), with_kwargs=True))
    hooks.append(layer.register_forward_hook(make_state_hook(li), with_kwargs=True))

press = ChunkedPress(compression_ratio=RATIO, block_length=B, mode="replace")
torch.manual_seed(1)
ids = torch.randint(1000, 20000, (1, L), device="cuda")

with torch.no_grad():
    with press(model):
        model.model(input_ids=ids, past_key_values=DynamicCache(), output_attentions=True)

for h in hooks:
    h.remove()

# ---- compare ----
# chunks with no eviction: post-chunk state scores == full candidate scores
# chunks with eviction: sel_cap holds the PRE-eviction scores (full candidate columns)
phys_before = [min(c * B, K) for c in range(L // B)]
worst_abs, worst_rel, n_cmp = 0.0, 0.0, 0
bad = []
for c in range(L // B):
    for li in range(NL):
        if (c, li) in sel_cap:
            scores = sel_cap[(c, li)]
        elif (c, li) in state_cap:
            scores = state_cap[(c, li)]
        else:
            bad.append((c, li, "missing capture"))
            continue
        att = attn_cap.get((c, li))
        if att is None:
            bad.append((c, li, "missing attention"))
            continue
        phys = phys_before[c]
        q_rows = att[0]  # (Hq, b, kv)
        kv = q_rows.shape[-1]
        # causal-corrected mass: survivors visible to all b queries; block slot m to b-m
        mass = q_rows.sum(dim=1)  # (Hq, kv)
        counts = torch.cat([torch.full((phys,), float(B)), torch.arange(B, 0, -1).float()]).to(q_rows)
        ref = mass / counts[None, :]
        ref = ref.view(Hkv, G, kv).mean(dim=1)  # GQA group mean -> (Hkv, kv)
        if ref.shape != scores.shape:
            bad.append((c, li, f"shape {tuple(scores.shape)} vs {tuple(ref.shape)}"))
            continue
        d = (scores - ref).abs()
        mabs = d.max().item()
        mrel = (d / ref.abs().clamp_min(1e-6)).mean().item()
        worst_abs = max(worst_abs, mabs)
        worst_rel = max(worst_rel, mrel)
        n_cmp += 1
        if mabs > 0.03:
            bad.append((c, li, f"max_abs={mabs:.4f}"))

print(f"compared {n_cmp} (chunk, layer) pairs over ALL columns (survivors + block)")
print(f"worst max_abs={worst_abs:.4e}  worst mean_rel={worst_rel:.4e}")
for b_ in bad[:6]:
    print("  BAD:", b_)

if n_cmp != (L // B) * NL:
    print(f"FAILED coverage: compared {n_cmp}, want {(L // B) * NL} (all chunks x all layers)")
    sys.exit(1)
if bad:
    print(f"FAILED ({len(bad)} bad pairs)")
    sys.exit(1)
print("SCORING REFERENCE (post-eviction, full columns) PASSED")
