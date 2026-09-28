# 最终技术总结：分块在线 KV 压缩的 NVRTC 评分融合算子

日期：2026-09-28。本文件为项目最终统一口径，依据已保存原始数据和 `verified/verified_summary.json` 的
独立复核结果整理；与旧阶段报告冲突之处以本文为准。

## 1. 问题

本项目基于 NVIDIA kvpress commit `a13a1da`，实现并验证
**ChunkedOnlinePress**：把 prefill 切成 B=256 的块逐块 forward；每块用当前真实 query
对候选 KV 评分，按绝对预算 K=1024 物理淘汰；后续块使用压缩后的缓存。
这与上游 BlockPress 已说明的全量 forward 后分块压缩不同（相关讨论见
[issue #186](https://github.com/NVIDIA/kvpress/issues/186)）。本文只描述该原型与已测基座，
不对上游全部方法或当前版本作缺失功能的概括。

压缩本身有明确的精度取舍（见 §6）；本阶段的问题是：**评分链路自身的执行开销有多大、
能否用自定义 GPU 算子消除，以及消除后精度是否保持**。

## 2. 实现一个完整案例

按"profiling 定位热点 → NVRTC 双 kernel → 正确性与数值处理 → 接回模型 → 原始数据验证"
完成闭环：

1. **定位**：8K 在线 prefill 的 torch.profiler 显示评分链（因果掩码构建 → masked_fill →
   softmax → query 维求和 → GQA 均值 → counts 归一）为每 (chunk, layer) 一组的 ~10 个小
   kernel，共 1152 组调用，启动延迟主导；评分 GEMM（einsum）保留在 PyTorch，不重写 attention。
2. **实现**：CUDA Toolkit 11.6 的 NVRTC 运行时编译（本机无 MSVC；ctypes 加载
   nvrtc64_112_0.dll 与 nvcuda.dll，sm_86 cubin）。两个无 block-wide barrier 的 kernel（使用 warp 同步 shuffle）：
   - `row_max_sum`：每 warp 一行，warp shuffle 归约行最大值（只对因果可见列）与 exp 和；
   - `col_accum`：每线程一列，Kahan 补偿累加 256 行归一化概率到寄存器；
   - 末尾两个微小 torch 算子做组归约与 counts 除法（counts 张量按 (kept,B,T) 缓存）。
3. **正确性与数值处理**（调试过程叙述来自原阶段报告；原始日志见
   `data/correctness/`；早期调试过程未全部纳入公开证据，本文不再重复声称独立核验）：
   - 初版单 kernel 设计实测更慢（0.5×），重构为双 kernel 后转为 2× 以上加速；
   - `__shfl_down_sync` 归约只有 lane 0 持有全量结果的 bug（行和系统性偏大 ~1e-3），
     改 butterfly `__shfl_xor_sync` 后单元误差降至 ≤2.8e-9；
   - 无头 NVRTC 环境无 `INFINITY`（位模式替代）；顺序 fp32 累加 256 项引入 ~1e-3 舍入
     （Kahan + 精确 expf 解决）；
   - 语义契约：T = kept + B（press 中恒成立），本次只验证此等式范围；当前代码仅拒绝 T > kept+B，不能推广为任意更小 T；T > 2048 回退 reference
     并置 `last_fallback` 标志。
4. **接回模型**：press 增加 `fused_scoring` 开关（默认 False，reference 路径不动）；
   评测管线经 `chunked_replace_k1024_fused` 名称显式启用。
5. **原始数据验证**：见 §4。

## 3. 正确性（本包证据范围）

| 检查 | 结果 | 产物 |
| --- | --- | --- |
| 融合单元检查（随机/边界/adversarial/确定性/契约断言/回退标志） | 13/13，max_abs ≤ 2.8e-9 | `data/performance/test_fused_scoring.log` |
| 门禁 G1–G6（融合路径全跑） | 全过 | `data/correctness/validate_fused.log` |
| 评分独立参考（144 对 vs eager 注意力，融合路径） | worst max_abs ≈ 5.2e-3（另一项检查，非 CUDA-vs-PyTorch 子链误差） | `data/correctness/scoring_ref_fused.log` |
| mask/位置探针（144 次调用） | 全过 | `data/correctness/mask_probe_fused.log` |
| dev650 精度回归 | 逐样本得分 650/650 相同；13 任务指标全部一致；宏平均 36.39 → 36.39 | `verified/verified_summary.json`、`data/*/metrics.json` |

**预测文本口径（修正后）**：633/650 条预测文本完全相同（97.38%），17 条文本不同
（multiquery 7、multivalue 5、single_1 2、single_3 2、multikey_3 1），但按官方 RULER
评分规则逐样本重算**零分差**。旧报告"650/650 逐字节一致"的说法撤回——那来自旧比较脚本
把简化正确性布尔值当作文本比较的误判。17 条文本差异的具体来源（如 top-k 近平手翻转）
本包无逐样本保留集合或 logits 证据，不作归因断言。

## 4. 性能收益（原始数据）

**微基准**（H=8、G=4、B=256、fp32，预热 5、50 次中位；被替换子链，不含 einsum）：

| T | Reference μs | Fused μs | 加速 |
|---|---:|---:|---:|
| 512 | 428.5 | 202.7 | 2.11× |
| 768 | 415.7 | 203.8 | 2.04× |
| 1024 | 481.3 | 226.3 | 2.13× |
| 1280 | 550.4 / 复测 547.8 | 241.7 | 2.28× / 2.27× |
| 2048 | 758.7 | 308.2 | 2.46× |

含评分 einsum（d=128、T=1280，该脚本 30 次重复）：749.1 → 556.0 μs，约 1.35×。

**端到端**（8K/16K/32K 合成输入；每长度两侧各预热 1 次；计时顺序为 ref→fused 重复 4 对，
非 AB/BA 平衡设计，也未做显著性检验；CUDA events，固定 32 步 decode）：

| 长度 | Reference prefill 中位 | Fused prefill 中位 | 中位下降 | fused 更快配对 |
|---|---:|---:|---:|---:|
| 8192 | 5235.9 ms | 4988.9 ms | 4.72% | 3/4 |
| 16384 | 10575.7 ms | 10097.8 ms | 4.52% | 3/4 |
| 32768 | 21702.0 ms | 20407.6 ms | 5.96% | 4/4 |

对外表述：**已测配置下 prefill 中位耗时下降约 4.5%–6.0%**；单次测量有波动
（原始 24 条计时全部保留于 `data/performance/perf_fused_alt.json`），不声称每次运行均加速。
峰值 allocated 显存 15.517 → 15.512 GiB，**基本持平**；decode 与 TTFT 的首步尾差在本实现
中未被优化，不据此宣称收益。注意 `kv_tokens=1056` 的读数是 benchmark 完成 32 步 decode
之后的缓存长度（= K + 32），不是纯 prefill 后的 K=1024 预算值；总 TTFT 包含 prefill，
"首 decode 尾步近似不变"不能写成"总 TTFT 不变"。

**Profiler**（8K 单次采集，用于解释实现，不替代端到端计时）：完整表脚 Self CUDA total
**4.094s → 3.894s，约 −4.89%**（Self CPU 5.773s → 5.488s）。旧报告 7954→7422ms 的口径
（对 top-40 行求和）把 aten 算子行与其子 kernel 行的设备时间重复累计，已撤回；报告
softmax 收益时也不应把 `aten::_softmax` 与其 CUDA kernel 时间相加（同一工作量的两种
归因视图）。逐算子行结论有效：masked_fill（1152 次调用合计约 119.6ms）与 softmax 系 kernel 归零。
历史包中另有一份 3.937s 的旧采集，与本包 4.094s 不是同一文件，不交叉混用。

## 5. 支持范围与 fallback

- batch=1；Qwen3/Llama 类注意力路径；标准"空缓存单次 prefill、问题与生成在 press 上下文外"
  的 pipeline 用法；其他接口边界（同一 press 上下文多次调用、显式位置首调用）未覆盖。
- 运行时为 Windows NVRTC + 驱动 API（固定 CUDA DLL 路径配置）；未验证 Linux、多 GPU、
  手机/嵌入式、多模型。
- `fused_scoring=False` 为默认，reference 链路完整保留；T > 2048 等超范围输入自动回退。
- 契约按实测范围表述为 **T = kept + B**（reference counts 的构造即按此等式）；不做
  T ≤ kept + B 的推广声称。

## 6. 局限与边界

- **压缩方法本身的精度损失依然存在**：同一 dev 子集 no-press 77.25、K=1024 在线压缩
  36.39。CUDA 优化保持的是压缩基线的得分，不解决评分信号质量，不能宣称相对完整模型无损。
- 在线 vs 匹配 oracle 的 +1.80pp 是固定子集观察，无显著性或等价性检验。
- 精度结论限定于该 650 条固定子集与当前模型/设备/配置；预测文本存在 17 条差异，
  不称位级等价；无逐样本 KV 保留集合差异的直接证据。
- 更早的 32K"压缩 vs 不压缩"对比（23.676s / 20.208s 等）属于压缩方法实验，计时入口与
  本轮 reference/fused 配对实验不同，不混入本轮结论。

## 7. 结论

在已测范围内，NVRTC 评分融合算子以零精度回归（逐样本得分与全部任务指标一致、
633/650 文本相同）换取评分子链 2.0–2.5×、端到端 prefill 约 4.5%–6.0% 的中位加速，
显存基本持平，且保留完整 reference 回退。该案例完成技术验收；预算 sweep、评分信号改进
（dev 上在线 36.39 vs no-press 77.25 的差距是方法层面的开放问题）、多模型、其他设备与
上游贡献为后续独立里程碑。
