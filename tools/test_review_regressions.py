"""Regression tests for the three P1 defects found in external review (REVIEW.md 2026-09-27).

R1  eviction remap must use the FULL candidate list as searchsorted base, and the
    physical KV content must equal the recorded original-token ids (single head).
R1b same, two heads with DIFFERENT per-head survivor sets (also catches head mixing
    in the online eviction gather).
R2  oracle finalize gather must preserve each KV head's own selection.
R3  prefill forwards must pass PHYSICAL cache_position and ORIGINAL position_ids
    (separated); transformers 5.2.0 mask contract verified against real installed helpers.

CPU-only. Exits non-zero on any failure. Logs bind the exact source file + sha256.
"""
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import transformers  # noqa: E402

import presses.chunked_online_press as P  # noqa: E402

SRC = Path(P.__file__)
SHA = hashlib.sha256(SRC.read_bytes()).hexdigest()

print(f"press source : {SRC}")
print(f"sha256       : {SHA}")
print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__} ({Path(transformers.__file__).as_posix()})")
assert SRC.resolve().is_relative_to(ROOT.resolve()), f"press module not from this tree: {SRC}"

FAILURES = []


def check(name, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------- #
# monkeypatch model-dependent helpers with identity stubs (same trick as the
# review reproduction, but on the REAL imported module)
# --------------------------------------------------------------------- #
P.extract_keys_and_values = lambda cache, li: (cache.layers[li].keys, cache.layers[li].values)
P.get_prerope_query_states = lambda module, hidden: module.review_queries


class FakeLayer:
    def __init__(self, keys):
        self.keys = keys
        self.values = keys.clone()


class FakeCache:
    def __init__(self, keys):
        self.layers = [FakeLayer(keys)]

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


def make_press(**kw):
    p = P.ChunkedOnlinePress(compression_ratio=0.5, block_length=2, n_sink=1, mode="replace", **kw)
    p._active = True
    return p


def run_hook(p, cache, module, n_new=2):
    hidden = torch.zeros(1, n_new, 4)
    pos = (torch.ones(1, n_new, module.head_dim), torch.zeros(1, n_new, module.head_dim))
    p._score_hook(module, [], {"hidden_states": hidden, "past_key_values": cache,
                               "position_embeddings": pos}, None)


def make_module(Hkv, Hq, b, d):
    """Attention-module stub: distinct per-head queries -> distinct scores -> deterministic selection."""
    m = type("M", (), {})()
    m.config = type("C", (), {"num_key_value_heads": Hkv, "num_attention_heads": Hq})()
    m.head_dim = d
    m.layer_idx = 0
    # per-head distinct queries (RoPE here is identity: cos=1, sin=0)
    torch.manual_seed(0)
    m.review_queries = torch.randn(1, Hq, b, d)
    return m


# --------------------------------------------------------------------- #
def r1_single_head_content():
    """Bug 1: physical KV content must equal recorded original-token ids (1 KV head)."""
    # survivors = originals [0,1,4,5]; current chunk = originals [6,7]; K=4
    p = make_press()
    p._state = {0: {"idx": torch.tensor([[0, 1, 4, 5]]), "scores": torch.ones(1, 4)}}
    p._cur = (6, 8)
    p._chunk_idx = 3
    p._K = 4

    cand_ids = torch.tensor([[0, 1, 4, 5, 6, 7]])
    sent = cand_ids.float().view(1, 1, 6, 1).expand(1, 1, 6, 2).clone()  # slot s stores token id s
    cache = FakeCache(sent)
    module = make_module(Hkv=1, Hq=1, b=2, d=2)

    run_hook(p, cache, module)
    selected = p._state[0]["idx"]
    actual = cache.layers[0].values[0, :, :, 0].long()
    check("R1.content_matches_record",
          torch.equal(selected, actual),
          f"recorded={selected.tolist()} actual={actual.tolist()}")


def r1b_two_head_content():
    """Bug 1 + 2 (online path): two heads, different per-head survivor sets, per-head content check."""
    # head0 survivors = [0,1,4,5]; head1 survivors = [1,3,4,5]; chunk = [6,7]; K=4
    p = make_press()
    idx = torch.tensor([[0, 1, 4, 5], [1, 3, 4, 5]])
    p._state = {0: {"idx": idx, "scores": torch.ones(2, 4)}}
    p._cur = (6, 8)
    p._chunk_idx = 3
    p._K = 4

    # physical layout: slots 0..3 hold per-head survivors, slots 4,5 hold the chunk
    phys = torch.zeros(1, 2, 6, 2)
    phys[0, 0, :, 0] = torch.tensor([0.0, 1, 4, 5, 6, 7])
    phys[0, 1, :, 0] = torch.tensor([1.0, 3, 4, 5, 6, 7])
    phys[..., 1] = 100.0  # head marker so cross-head copies are detectable
    cache = FakeCache(phys)
    module = make_module(Hkv=2, Hq=2, b=2, d=2)

    run_hook(p, cache, module)
    selected = p._state[0]["idx"]                      # (2, K) original ids
    actual = cache.layers[0].values[0, :, :, 0].long()  # (2, K)
    check("R1b.content_matches_record_per_head", torch.equal(selected, actual),
          f"recorded={selected.tolist()} actual={actual.tolist()}")
    check("R1b.no_cross_head_marker", bool((cache.layers[0].values[0, :, :, 1] == 100.0).all()),
          "head marker column must stay intact")


def r2_oracle_head_gather():
    """Bug 2: oracle finalize must preserve each KV head's own selection."""
    p = make_press(oracle_feedback=True)
    selected = torch.tensor([[0, 2, 4], [1, 3, 5]])
    p._state = {0: {"idx": selected}}
    src = torch.zeros(1, 2, 6, 1)
    src[0, 0, :, 0] = torch.arange(6, dtype=torch.float32)
    src[0, 1, :, 0] = torch.arange(100, 106, dtype=torch.float32)
    cache = FakeCache(src)
    p._finalize_oracle(cache)
    expected = src.gather(2, selected[None, :, :, None])
    check("R2.oracle_per_head_gather", torch.equal(expected, cache.layers[0].values),
          f"expected={expected[0,:,:,0].tolist()} actual={cache.layers[0].values[0,:,:,0].tolist()}")


def r3_prefill_positions():
    """Bug 3: chunked prefill must pass PHYSICAL cache_position and ORIGINAL position_ids."""
    L, B, K = 1024, 256, 512
    p = make_press()
    p.block_length = B
    recorded = []

    def fake_forward(model_self=None, input_ids=None, past_key_values=None, **kw):
        recorded.append({
            "cache_position": kw.get("cache_position"),
            "position_ids": kw.get("position_ids"),
        })
        b = input_ids.shape[1]
        layer = past_key_values.layers[0]
        layer.keys = torch.cat([layer.keys, torch.zeros(1, 1, b, 2)], dim=2)  # simulate append
        if layer.keys.shape[2] > p._K:  # simulate the hook's eviction back to budget K
            layer.keys = layer.keys[:, :, : p._K]
        return "OUT"

    fake_cache = FakeCache(torch.zeros(1, 1, 0, 2))
    out = p._chunked_prefill(fake_forward, None, torch.zeros(1, L, dtype=torch.long),
                             {"past_key_values": fake_cache})
    check("R3.returns_last_output", out == "OUT")

    exp_phys = [min(c * B, K) for c in range(L // B)]
    ok_cp, ok_pid = True, True
    for c, rec in enumerate(recorded):
        s = c * B
        want_cp = torch.arange(exp_phys[c], exp_phys[c] + B)
        want_pid = torch.arange(s, s + B).unsqueeze(0)
        got_cp = rec["cache_position"]
        got_pid = rec["position_ids"]
        if got_cp is None or not torch.equal(got_cp.cpu(), want_cp):
            ok_cp = False
            print(f"    chunk {c}: cache_position got={None if got_cp is None else got_cp.tolist()[:4]}... want={want_cp.tolist()[:4]}...")
        if got_pid is None or not torch.equal(got_pid.cpu(), want_pid):
            ok_pid = False
            print(f"    chunk {c}: position_ids got={None if got_pid is None else got_pid.tolist()[0][:4]}... want={want_pid.tolist()[0][:4]}...")
    check("R3.cache_position_physical", ok_cp, f"(physical starts {exp_phys})")
    check("R3.position_ids_original", ok_pid, "(original absolute positions for RoPE)")


def r3b_transformers_mask_contract():
    """The fix relies on: mask = (physical slot <= cache_position), sizes from real cache length.
    Verified against the INSTALLED transformers 5.2.0 helpers (version printed above)."""
    from transformers.cache_utils import DynamicLayer
    from transformers.masking_utils import sdpa_mask

    layer = DynamicLayer()
    layer.is_initialized = True  # reviewer's repro does the same; else get_seq_length() returns 0
    layer.keys = torch.zeros(1, 1, 512, 1)  # post-eviction physical length
    cache_position = torch.arange(512, 768)  # what fix C passes for chunk 3
    kv_length, kv_offset = layer.get_mask_sizes(cache_position)
    mask = sdpa_mask(1, cache_position, kv_length, kv_offset, allow_is_causal_skip=False)[0, 0]  # (q, kv) bool

    want = (torch.arange(768)[None, :] <= (512 + torch.arange(256))[:, None])
    check("R3b.mask_sizes", kv_length == 768 and kv_offset == 0, f"kv_length={kv_length} offset={kv_offset}")
    check("R3b.first_query_sees_513", int(mask[0].sum()) == 513, f"visible={int(mask[0].sum())}")
    check("R3b.full_pattern", torch.equal(mask, want), "rows = survivors + intra-block causal")


if __name__ == "__main__":
    r1_single_head_content()
    r1b_two_head_content()
    r2_oracle_head_gather()
    r3_prefill_positions()
    r3b_transformers_mask_contract()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL REGRESSIONS PASSED")
