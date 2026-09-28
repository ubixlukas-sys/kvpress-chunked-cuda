# SPDX-License-Identifier: Apache-2.0
"""ChunkedOnlinePress: true chunked-prefill online KV compression for kvpress.

Design contract: D:/mxy/llm4/stage2_1_spec.md (v1).

- The full-sequence prefill forward is intercepted (KVzipPress-style wrapped
  ``model.model.forward``) and re-executed block by block.
- After each block, a per-layer hook scores every candidate KV slot (survivors
  + current block) with the current block's real queries: causal-masked
  attention mass normalized by the number of queries that can see the slot.
- Score update rules (the ONLY difference between modes):
    replace: s <- a                          (all slots, every block)
    ema:     old: s <- g*s+(1-g)*a; new: s <- a
    sum:     old: s <- s+a;         new: s <- a     (pure accumulation)
- Budget K: `budget` (absolute per-layer KV tokens) or `compression_ratio`
  (K = max(1, int(L*(1-ratio)))); exactly one of them may be set. K includes sinks +
  survivors + current-block survivors. Transient peak per block forward is
  K + block_length by design.
- Position semantics: model forwards receive PHYSICAL `cache_position`
  (arange(phys_len, phys_len+b), drives causal mask + cache append) and ORIGINAL
  `position_ids` (drives RoPE only). Kept KVs thus keep their original rotary
  positions (gapped, kvpress convention) while the mask stays causal over the
  compacted physical layout.
- oracle_feedback=True: identical scoring/budget/eviction bookkeeping, but no
  physical eviction in-loop (hidden states come from the full history); the
  final selection is applied physically once at the end. Quantifies the cost
  of the online constraint.
- strict_checks=True verifies exact-match on the eviction remap each time it runs
  (cheap tensor assert, forces a GPU sync; disable for pure performance runs).

Limitations: batch 1; Llama/Qwen3-like attention; no attention_mask support;
kept KVs keep original RoPE positions (gapped positions, kvpress convention).
"""

import math
import types
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
from kvpress.utils import compute_n_kept, extract_keys_and_values, get_prerope_query_states


@dataclass
class ChunkedOnlinePress(BasePress):
    compression_ratio: float = 0.0
    budget: int | None = None  # absolute per-layer KV budget (wins over ratio; ratio must stay 0.0)
    block_length: int = 256
    n_sink: int = 4
    mode: str = "ema"  # "ema" | "replace" | "sum"
    gamma: float = 0.9
    oracle_feedback: bool = False
    strict_checks: bool = True  # exact-match verification on eviction remap (forces a GPU sync; disable for perf runs)
    fused_scoring: bool = False  # NVRTC fused mask+softmax+query-sum kernel (reference chain kept as fallback)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1
        if self.budget is not None:
            assert self.budget >= 1
            assert self.compression_ratio == 0.0, "provide either budget or compression_ratio, not both"
        assert self.block_length > 0
        assert self.n_sink >= 1, "at least one sink token required"
        assert self.mode in ("ema", "replace", "sum"), f"unknown mode: {self.mode}"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"
        self._active = False
        self._state = {}  # layer_idx -> {"idx": (Hkv,T) int64, "scores": (Hkv,T) fp32}
        self._cur = (0, 0)  # current chunk [start, end) in ORIGINAL token positions
        self._chunk_idx = 0
        self._K = 0
        self._sim_prev = {}  # per-layer simulated (== physical, non-oracle) length before current chunk
        self._transient_lens = []

    # ------------------------------------------------------------------ #
    @contextmanager
    def __call__(self, model: PreTrainedModel):
        if not isinstance(model, SUPPORTED_MODELS):
            import logging

            logging.getLogger(__name__).warning(f"Model {type(model)} not tested: {SUPPORTED_MODELS}")
        self.reset()
        self._active = True
        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self._score_hook, with_kwargs=True))

            original_forward = model.model.forward
            press = self

            def wrapped_forward(model_self, *args, **kwargs):
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                if input_ids is None:
                    return original_forward(*args, **kwargs)
                # Only fresh prefill calls are intercepted. Question/decode passes
                # (kvpress pipeline) carry explicit position_ids / cache_position /
                # inputs_embeds and must go through untouched.
                if (kwargs.get("position_ids") is not None
                        or kwargs.get("cache_position") is not None
                        or kwargs.get("inputs_embeds") is not None):
                    return original_forward(*args, **kwargs)
                if kwargs.get("attention_mask", None) is not None:
                    raise NotImplementedError("ChunkedOnlinePress does not support attention_mask yet")
                L = input_ids.shape[1]
                if L <= press.block_length:
                    # single chunk, still route through the scoring path
                    cache = kwargs.get("past_key_values")
                    if cache is None:
                        cache = DynamicCache()
                        kwargs["past_key_values"] = cache
                    press._begin(L, input_ids.device)
                    phys_start = cache.get_seq_length()
                    press._cur = (0, L)
                    out = original_forward(
                        input_ids=input_ids,
                        past_key_values=cache,
                        cache_position=torch.arange(phys_start, phys_start + L, device=input_ids.device),
                        position_ids=torch.arange(0, L, device=input_ids.device).unsqueeze(0),
                        use_cache=True,
                    )
                    press._chunk_idx += 1
                    if press.oracle_feedback:
                        press._finalize_oracle(cache)
                    press._active = False
                    return out
                return press._chunked_prefill(original_forward, model_self, input_ids, kwargs)

            model.model.forward = types.MethodType(wrapped_forward, model.model)
            try:
                yield
            finally:
                model.model.forward = original_forward
        finally:
            for h in hooks:
                h.remove()
            self._active = False

    # ------------------------------------------------------------------ #
    def _begin(self, L: int, device):
        self._K = self.budget if self.budget is not None else compute_n_kept(L, self.compression_ratio)
        self._chunk_idx = 0
        self._sim_prev = {}
        self._transient_lens = []
        self._state = {}

    def _chunked_prefill(self, original_forward, model_self, input_ids, kwargs):
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = DynamicCache()
            kwargs["past_key_values"] = cache
        L = input_ids.shape[1]
        device = input_ids.device
        self._begin(L, device)
        out = None
        for start in range(0, L, self.block_length):
            end = min(start + self.block_length, L)
            b = end - start
            self._cur = (start, end)
            # PHYSICAL cache_position drives the model's causal mask and cache append
            # (DynamicLayer mask sizes = physical length + q_len, causal compares slot
            # <= cache_position); ORIGINAL positions drive RoPE only.
            phys_start = cache.get_seq_length()
            cache_position = torch.arange(phys_start, phys_start + b, device=device)
            position_ids = torch.arange(start, end, device=device).unsqueeze(0)
            with torch.no_grad():
                out = original_forward(
                    input_ids=input_ids[:, start:end],
                    past_key_values=cache,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    use_cache=True,
                )
            self._chunk_idx += 1
        if self.oracle_feedback:
            self._finalize_oracle(cache)
        self._active = False
        return out

    # ------------------------------------------------------------------ #
    def _score_hook(self, module: nn.Module, input: list, kwargs: dict, output: list):
        if not self._active:
            return output
        bsz = kwargs["hidden_states"].shape[0]
        assert bsz == 1, "ChunkedOnlinePress supports batch size 1 only"

        with torch.no_grad():
            cache = kwargs["past_key_values"]
            layer_idx = int(module.layer_idx)
            keys, values = extract_keys_and_values(cache, layer_idx)
            Hkv = module.config.num_key_value_heads
            Hq = module.config.num_attention_heads
            G = Hq // Hkv
            d = module.head_dim
            dev = keys.device

            T_phys = keys.shape[2]
            b = self._cur[1] - self._cur[0]
            self._transient_lens.append(T_phys)

            st = self._state.get(layer_idx)
            if st is None:
                sim_len = 0
                idx = None  # (Hkv, T_sim)
                scores = None
            else:
                sim_len = st["idx"].shape[-1]
                idx, scores = st["idx"], st["scores"]

            # candidate new-part indices are ORIGINAL token positions of this chunk,
            # not physical slots (physical slots renumber after each compaction)
            chunk_orig = torch.arange(self._cur[0], self._cur[1], device=dev)

            # candidate list = simulated survivors + current block (ascending indices)
            if idx is None:
                cand_idx = chunk_orig.view(1, -1).expand(Hkv, -1).contiguous()  # (Hkv, b)
            else:
                cand_idx = torch.cat(
                    [idx, chunk_orig.view(1, 1, -1).expand(Hkv, 1, -1).reshape(Hkv, -1)], dim=-1
                )  # (Hkv, T_sc)

            # ---- scoring: current block queries vs candidates ----
            if self.oracle_feedback and idx is not None:
                # oracle: physical layout is uncompressed so slot == original index
                gather_idx = cand_idx[None, :, :, None].expand(1, Hkv, -1, d)
                k_sc = keys.gather(2, gather_idx)
                kept_for_mask = sim_len
            else:
                k_sc = keys  # contiguous: [kept in idx order..., current chunk...]
                kept_for_mask = sim_len if st is not None else 0

            q = get_prerope_query_states(module, kwargs["hidden_states"])  # (1,Hq,b,d) pre-RoPE
            cos, sin = kwargs["position_embeddings"]
            q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            q = q.view(1, Hkv, G, b, d).float()
            k = k_sc.view(1, Hkv, -1, d).float()
            T_sc = k.shape[2]

            logits = torch.einsum("bhgqd,bhkd->bhgqk", q, k) / math.sqrt(d)  # (1,Hkv,G,b,T_sc)
            n_old = T_sc - b
            if self.fused_scoring:
                # NVRTC fused mask+softmax+query-sum kernel; contract T <= kept+b
                # holds by construction. Falls back to the reference chain on
                # non-CUDA / oversized T (recorded via fused_scoring.last_fallback).
                from presses.fused_scoring import fused_scoring_mass

                a_norm = fused_scoring_mass(logits, kept_for_mask, b)
            else:
                # reference chain (frozen implementation, kept as fallback)
                jpos = torch.arange(b, device=dev).view(b, 1)
                mpos = torch.arange(T_sc, device=dev).view(1, T_sc)
                allowed = mpos <= kept_for_mask + jpos  # (b, T_sc)
                logits = logits.masked_fill(~allowed.view(1, 1, 1, b, T_sc), float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                mass = probs.sum(dim=-2).mean(dim=2)[0]  # (Hkv, T_sc): sum over queries, mean over GQA group

                counts = torch.cat(
                    [torch.full((n_old,), float(b), device=dev), torch.arange(b, 0, -1, device=dev).float()]
                ).view(1, T_sc)
                a_norm = mass / counts  # causal-corrected average attention per query

            # ---- score update ----
            if bool((cand_idx[:, 1:] <= cand_idx[:, :-1]).any()):
                raise RuntimeError(
                    f"candidate indices not strictly ascending at layer {layer_idx} chunk {self._chunk_idx}"
                )
            if scores is None:
                scores = a_norm.clone()
            else:
                old, new = a_norm[:, :n_old], a_norm[:, n_old:]
                if self.mode == "replace":
                    scores = torch.cat([old, new], dim=-1)
                elif self.mode == "ema":
                    scores = torch.cat([self.gamma * scores + (1 - self.gamma) * old, new], dim=-1)
                else:  # sum
                    scores = torch.cat([scores + old, new], dim=-1)

            # ---- eviction ----
            W = min(self.block_length, max(0, self._K - self.n_sink))
            evicted = cand_idx.shape[-1] > self._K
            if evicted:
                idx_new, scores_new = self._select(scores, cand_idx, self._K, self.n_sink, W)
            else:
                idx_new, scores_new = cand_idx, scores

            if not self.oracle_feedback:
                if evicted:
                    # idx_new holds ORIGINAL token indices. Physical layout: survivors
                    # in ascending-original order (== idx rows), then the current block,
                    # so the physical slot of a candidate is its position in cand_idx.
                    # The searchsorted base MUST be the full candidate list: using only
                    # idx maps every current-block index to len(idx) (duplicate slots).
                    cand_sorted = cand_idx.contiguous()
                    pos = torch.searchsorted(cand_sorted, idx_new.contiguous())
                    if self.strict_checks:
                        assert bool((pos < cand_sorted.shape[-1]).all()), (
                            f"remap: selected index missing from candidates at layer {layer_idx} "
                            f"chunk {self._chunk_idx}"
                        )
                        assert torch.equal(cand_sorted.gather(1, pos), idx_new.contiguous()), (
                            f"remap: exact-match verification failed at layer {layer_idx} "
                            f"chunk {self._chunk_idx}"
                        )
                    g = pos[None, :, :, None].expand(1, Hkv, -1, d)
                    cache.layers[layer_idx].keys = keys.gather(2, g)
                    cache.layers[layer_idx].values = values.gather(2, g)
                # else: no eviction -> physical layout already equals cand order
                self._sim_prev[layer_idx] = idx_new.shape[-1]
            else:
                self._sim_prev[layer_idx] = idx_new.shape[-1]

            self._state[layer_idx] = {"idx": idx_new.contiguous(), "scores": scores_new.contiguous()}
        return output

    @staticmethod
    def _select(scores: torch.Tensor, cand_idx: torch.Tensor, K: int, n_sink: int, W: int = 0):
        """Protect sinks (smallest S indices) + the recent window (largest W indices);
        the middle competes by score for the remaining K - S - W slots; result sorted."""
        S = min(n_sink, K)
        W = min(W, max(0, K - S))
        T = cand_idx.shape[-1]
        tail_start = max(S, T - W)
        head = cand_idx[:, :S]
        mid = cand_idx[:, S:tail_start]
        tail = cand_idx[:, tail_start:]
        n_mid_keep = min(max(0, K - S - W), mid.shape[-1])
        top = scores[:, S:tail_start].topk(n_mid_keep, dim=-1).indices
        keep_mid = torch.gather(mid, 1, top)
        keep = torch.cat([head, keep_mid, tail], dim=1).sort(dim=-1).values.contiguous()
        keep_scores = torch.gather(scores, 1, torch.searchsorted(cand_idx.contiguous(), keep))
        return keep, keep_scores

    # ------------------------------------------------------------------ #
    def _finalize_oracle(self, cache):
        for layer_idx, st in self._state.items():
            idx = st["idx"]  # (Hkv, T_sim) original indices, per head
            layer = cache.layers[layer_idx]
            Hkv, T_sim = idx.shape
            d = layer.keys.shape[-1]
            g = idx[None, :, :, None].expand(1, Hkv, T_sim, d)
            layer.keys = layer.keys.gather(2, g)
            layer.values = layer.values.gather(2, g)

    def reset(self):
        self._active = False
        self._state = {}
        self._cur = (0, 0)
        self._chunk_idx = 0
        self._K = 0
        self._sim_prev = {}
        self._transient_lens = []


def register_chunked_presses(registry: dict, name: str) -> list:
    """Register a ChunkedOnlinePress under `name` in a PRESS_REGISTRY-like dict.

    Name grammar: chunked_<mode>[_b<block>][_k<budget>][_oracle][_fused], e.g.
    chunked_ema, chunked_replace, chunked_sum, chunked_ema_b512, chunked_ema_k1024,
    chunked_ema_oracle. _k sets the absolute per-layer budget (compression_ratio must
    then be left at 0); without _k the budget derives from the press compression_ratio.
    _fused enables the NVRTC fused scoring kernel (reference chain kept as fallback).
    """
    parts = name.split("_")
    assert parts[0] == "chunked" and len(parts) >= 2, f"bad press name: {name}"
    mode = parts[1]
    assert mode in ("ema", "replace", "sum"), f"bad mode in name: {name}"
    kwargs = {}
    for p in parts[2:]:
        if p.startswith("b") and p[1:].isdigit():
            kwargs["block_length"] = int(p[1:])
        elif p.startswith("k") and p[1:].isdigit():
            kwargs["budget"] = int(p[1:])
        elif p == "oracle":
            kwargs["oracle_feedback"] = True
        elif p == "fused":
            kwargs["fused_scoring"] = True
        else:
            raise ValueError(f"unknown name part: {p} in {name}")
    registry[name] = ChunkedOnlinePress(mode=mode, **kwargs)
    return [name]
