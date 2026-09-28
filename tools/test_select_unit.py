import sys

sys.path.insert(0, r"D:\mxy\llm4")
import torch

from presses.chunked_online_press import ChunkedOnlinePress

out = []
torch.manual_seed(0)
Hkv, T, K, S = 8, 768, 512, 4
cand = torch.arange(T).view(1, -1).expand(Hkv, -1).contiguous()
scores = torch.rand(Hkv, T)
keep, keep_scores = ChunkedOnlinePress._select(scores, cand, K, S)
out.append(f"keep shape: {tuple(keep.shape)}")
out.append(f"ascending: {bool((keep[:, 1:] > keep[:, :-1]).all())}")
out.append(f"sink protected: {torch.equal(keep[:, :S], torch.arange(S).expand(Hkv, -1))}")
manual = torch.gather(scores, 1, keep)
out.append(f"scores aligned: {torch.equal(manual, keep_scores)}")
# 再测首轮（T=K 以下不淘汰）与重复调用稳定性
keep2, ks2 = ChunkedOnlinePress._select(scores, cand, K, S)
out.append(f"deterministic: {torch.equal(keep, keep2)}")
print("\n".join(out), flush=True)
