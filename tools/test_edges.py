# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for _select and the chunked path (spec audit item 4).

Covers: K < n_sink, K=1, short sequence (L < B), incomplete last block,
protected-set dedup (head/tail/mid disjoint), searchsorted exact match.
"""

import sys

HERE = r"D:\mxy\llm4"
sys.path.insert(0, HERE)

import torch  # noqa: E402

from presses.chunked_online_press import ChunkedOnlinePress  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        FAILS.append(name)


def sel(scores, cand, K, S, W):
    return ChunkedOnlinePress._select(scores, cand, K, S, W)


def verify_contract(scores, cand, K, S, W, keep, keep_scores, tag):
    T = cand.shape[-1]
    check(f"{tag}.size_eq_K", keep.shape[-1] == min(K, T), f"keep={keep.shape[-1]} K={K} T={T}")
    check(f"{tag}.ascending", bool((keep[:, 1:] > keep[:, :-1]).all()))
    check(f"{tag}.subset", bool(torch.isin(keep, cand).all()))
    # searchsorted exact match: scores at keep == keep_scores
    manual = torch.gather(scores, 1, keep)
    check(f"{tag}.scores_aligned", torch.equal(manual, keep_scores))
    # dedup: protected head/tail never appear twice
    check(f"{tag}.unique", bool((keep[:, 1:] != keep[:, :-1]).all()))


def main():
    torch.manual_seed(0)
    Hkv = 8

    # ---- E1: K < n_sink (budget smaller than sink count) ----
    T = 100
    cand = torch.arange(T).view(1, -1).expand(Hkv, -1).contiguous()
    scores = torch.rand(Hkv, T)
    K, S, W = 2, 4, 0
    keep, ks = sel(scores, cand, K, S, W)
    verify_contract(scores, cand, K, S, W, keep, ks, "E1.K_lt_S")
    check("E1.keeps_first_K", torch.equal(keep, cand[:, :K]), "expect first K originals")

    # ---- E2: K == 1 ----
    keep, ks = sel(scores, cand, 1, 4, 0)
    verify_contract(scores, cand, 1, 4, 0, keep, ks, "E2.K_eq_1")
    check("E2.keeps_first_1", torch.equal(keep, cand[:, :1]))

    # ---- E3: W protection with normal budget ----
    K, S, W = 50, 4, 10
    keep, ks = sel(scores, cand, K, S, W)
    verify_contract(scores, cand, K, S, W, keep, ks, "E3.W_protect")
    head_ok = torch.equal(keep[:, :S], cand[:, :S])
    tail_set = set(cand[0, -W:].tolist())
    kept_set = set(keep[0].tolist())
    check("E3.sink_kept", head_ok)
    check("E3.tail_kept", tail_set.issubset(kept_set))

    # ---- E4: W larger than budget allows (clamped) ----
    keep, ks = sel(scores, cand, 20, 4, 100)
    verify_contract(scores, cand, 20, 4, 100, keep, ks, "E4.W_clamp")
    check("E4.size", keep.shape[-1] == 20)

    # ---- E5: T == K (no-op boundary) ----
    keep, ks = sel(scores, cand, T, 4, 10)
    verify_contract(scores, cand, T, 4, 10, keep, ks, "E5.T_eq_K")
    check("E5.keeps_all", torch.equal(keep, cand))

    # ---- E6: press-level edge: L < B and L == B (single chunk path) ----
    import os

    os.environ.setdefault("HF_HOME", os.path.join(HERE, "cache", "hf"))
    import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

    sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
    from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

    from presses.chunked_online_press import compute_n_kept  # noqa: E402

    model = AutoModelForCausalLM.from_pretrained(
        r"D:\mxy\llm\kvpress_project\models\Qwen3-8B", dtype=torch.bfloat16
    ).to("cuda").eval()
    vocab = model.config.vocab_size

    for L in (64, 128, 300):  # < B, == B, > B with incomplete last block (300 = 256+44)
        ids = torch.randint(1000, vocab - 10, (1, L), device="cuda")
        press = ChunkedOnlinePress(compression_ratio=0.25, block_length=256)
        cache = DynamicCache()
        with torch.no_grad(), press(model):
            model(input_ids=ids, past_key_values=cache)
        Kexp = compute_n_kept(L, 0.25)
        lens = [cache.layers[i].keys.shape[2] for i in range(len(cache.layers))]
        check(f"E6.L{L}.final_len", all(x == Kexp for x in lens), f"lens={set(lens)} Kexp={Kexp}")
        asc = all(bool((press._state[i]["idx"][:, 1:] > press._state[i]["idx"][:, :-1]).all()) for i in range(len(cache.layers)))
        check(f"E6.L{L}.ascending", asc)

    print()
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL EDGE TESTS PASSED")


if __name__ == "__main__":
    main()
