# EXPlace 实验归档索引

本文件是历史运行的导航页。原始日志和 CSV 保留在各自目录中；这里记录运行目的、
状态、可比性和结论，避免只看一个 summary.csv 就误读实验。

## 运行目录

| 日期/目录 | 类型 | 状态 | 说明 |
| --- | --- | --- | --- |
| `long_runs/20260920_134832` | matrix dry-run | 计划生成 | 12 个任务的 `plan.tsv`；`master.log` 明确标记 `dry_run: 1`，没有训练结果，不应视为失败实验。 |
| `long_runs/20260920_140834` | 4 variant x 3 seeds | 12/12 成功 | `superblue18`，每任务 1500 episodes、150 条训练日志、15 次物理评估，单 GPU 串行约 11.02 小时。 |
| `comparison_runs/rl_logs/superblue18/20260920_130221` | quick smoke | 完成 | CPU、`compare_quick.yaml`，用于 step/global 流程检查，不用于 PPA 结论。 |
| `comparison_runs/rl_logs/superblue18/20260920_130336` | quick smoke | 完成 | CPU、短运行；检查配置和 rollout/update 路径。 |
| `comparison_runs/rl_logs/superblue18/20260920_130535` | quick timing | 完成 | step/global 对比数据及 TensorBoard/CSV。 |
| `comparison_runs/rl_logs/superblue18/20260920_132122` | quick timing | 完成 | step/global 对比数据及 TensorBoard/CSV。 |
| `comparison_runs/rl_logs/superblue18/20260920_132846` | quick timing | 完成 | step/global 对比数据及 TensorBoard/CSV。 |
| `comparison_runs/rl_logs/superblue18/20260920_133536` | quick timing | 完成 | step/global 对比数据及 TensorBoard/CSV。 |
| `optimization_runs/combo_long_20260924_061555` + `combo_quality_20260924_144146` | 性能/质量配对 | 15/15 + 15/15 成功 | 最快组合墙钟提升 4.1%，但所有组合都有不同程度 PPA 损失；轻量记录见 `experiment_records/20260924_optimization_combo/`。 |
| `long_runs/global_multidesign_legacy_20261008_111740` | 4 电路 reward 矩阵 | 48/48 成功，但 adaptive 无效 | 同步路径未调用 learner，adaptive 两臂分别等同 fixed；仅 fixed 臂可作基线。 |
| `long_runs/global_multidesign_adaptive_fixed_20261012_095649` | 修复后 adaptive 重跑 | 24/24 成功 | 4 电路 x 3 seeds x 2 adaptive 粒度；权重从 episode 400 开始更新，记录见 `experiment_records/202610_reward_multidesign/`。 |
| `long_runs/legalization_hannan_fix_20261020_093027` | 合法性短测 | 2/2 成功，10/10 合法 | superblue18 seed 3，验证 local-grid + Hannan fallback。 |
| `long_runs/legalization_matrix_20261020_100329` | 跨电路合法性矩阵 | **未完成：28/48 成功** | 另有 2 个任务中断，累计 132/145 物理评估合法；superblue4 仍有 13 次失败。轻量记录见 `experiment_records/20261020_legalization/`。 |
| `long_runs/20261027_105819_legalization_finish` | 24 任务合法性矩阵（50 ep） | 24/24 成功 | sb18 **60/60** 合法，sb4 **47/60**（78.3%）；13 次失败全为 `dreamplace_legality_check_failed`，形态是 std cell ↔ fixed macro 薄条，集中 seed 7。仅正确性验证，P1-1 不能用它读 PPA。轻量记录见 `experiment_records/20261027_legalization_finish/`。 |
| `long_runs/step_gamma1_20261108` | `step+gamma=1` 控制组 | 6/6 成功 | sb18，seeds 3/7/11，`fixed_step`+`adaptive_step`，`config/step_gamma1.yaml`（`reward_discount_gamma=1.0`，`gamma=0.95` 不动），1500 ep，双 L40S 并行约 4 小时。记录见 `experiment_records/20261108_superblue18_step_gamma1/`。 |

## 主要长跑结果

详细表格见 [`long_runs/20260920_140834/RESULT_ANALYSIS.md`](long_runs/20260920_140834/RESULT_ANALYSIS.md)。
可提交、可复算的派生记录见
[`experiment_records/20260920_superblue18_4x3/README.md`](experiment_records/20260920_superblue18_4x3/README.md)。

当前最强但有边界的结论：

- `adaptive_step` 的 best TNS 是四个 variant 中最好。
- `adaptive_global` 的 final GP-HPWL、final TNS、final WNS 和 best-to-final 稳定性最好。
- `fixed_global` 说明只改变 reward 粒度不能保证改善。
- **奖励粒度与折扣因子是两个独立因素**（`step+gamma=1` 控制组，2026-11-08 补齐）：
  - 折扣 0.95 → 1.0 使 `adaptive_step` best TNS 从 -133.61 改善到 **-123.88**；
  - 但 `step+gamma=1` 与 `global+gamma=1` 并不等价：前者峰值 TNS 最好，后者终值 TNS
    最好（-148.02）且最稳（GP 回归 0.08% vs 1.70%）；
  - 因此 H1 不能写成「只是折扣偏差」；正确的读法是 **step 适合冲峰值（配合
    backtrack 取 checkpoint），global 适合稳定收敛（用 final 模型）**。
- 三个 seed 都出现 `df` 上升、`reg` 下降，但 `port=.05` 是配置中的固定项，不能视为自然学习结果。
- 结果是 `superblue18`、3 seeds、DREAMPlace + OpenTimer post-placement 流程；不能直接替代 post-GRT/post-DRT 或论文 PPA signoff。

原始文件：

- `long_runs/20260920_140834/summary.csv`
- `long_runs/20260920_140834/master.log`
- `long_runs/20260920_140834/<variant>/superblue18/seed_<n>/output/rl_logs/.../eval_metrics.csv`
- `long_runs/20260920_140834/<variant>/superblue18/seed_<n>/output/rl_logs/.../reward_weights.csv`

这次长跑早于 metadata 自动采集功能，因此不存在同期 `run_metadata.txt`。保留的
`master.log` 能证明启动时选择了 `/usr/bin/python3` 和 GPU 0，DREAMPlace
`configure.py` 能证明参考扩展由 CUDA 11.8 构建；但这些证据不能还原当时的 GPU
型号和 driver。不得用当前主机快照回填并冒充历史同期 metadata。

## 已测效率结果

见 [`EFFICIENCY.md`](EFFICIENCY.md)。当前已确认：

- CPU quick smoke、2 Ray workers、4 episodes、无物理评估时，step/global 总时长约 23 秒。
- float32 observation 改动将同一 smoke run 从 23.34 秒降到 22.69 秒，约 2.8%。
- 长跑普通 batch 的 rollout 约 9.5 秒、PPO 约 8 秒；物理评估 batch 会明显变慢。
- 关闭额外 best-reward DREAMPlace 评估可以避免重复物理评估。

## 早期/诊断运行

以下日志用于定位环境和性能，不与正式矩阵混合：

- `test_run.log`：早期 DREAMPlace/OpenTimer 运行诊断。
- `test_superblue18_run.log`：superblue18 物理评估和 legalization 诊断。
- `test_quick_run.log`：短训练/配置 smoke。
- `comparison_step.log`：CPU quick step baseline。
- `comparison_step_float32.log`：float32 observation 对照。
- `long_runs/superblue18_3seed.launch.log`：一次启动环境/依赖诊断；出现过 `torch` 不可导入，不能作为算法失败记录。

## 归档规则

以后每次运行至少保存：

```text
run root:
purpose:
date/time:
command:
python path and version:
torch/CUDA/driver/GPU:
git commit and dirty files:
config/resolved_config:
benchmark/seeds/variants:
evaluation stage:
status and exit code:
raw log/CSV paths:
result summary:
known caveats:
next action:
```

矩阵脚本现在会自动运行 `script/capture_run_metadata.sh`，并将工具链快照写入
`<RUN_ROOT>/run_metadata.txt`。如果运行是在没有 driver/nvcc 的主机上进行，快照会
明确记录 unavailable；这类快照只能证明环境状态，不能证明 GPU 性能。

`script/summarize_experiment.py` 以 `gp_hpwl > 0` 识别实际 DREAMPlace 评估行，因为
未评估 batch 的 CSV 占位值是 0。它会拒绝同一 variant/benchmark/seed 下出现多个
`eval_metrics.csv`，避免静默混合恢复运行或重复目录。

CUDA 12 算子迁移另见 [`CUDA12_MIGRATION.md`](CUDA12_MIGRATION.md)，不应把未通过数值
和 placement-level gate 的构建写入正式结果表。

2026 年 9–10 月新增的轻量归档只保存配置计划、环境快照、任务状态和派生 CSV。
原始 checkpoint、DEF、TensorBoard 和完整运行目录总计数十 GB，继续留在本地并由
`.gitignore` 排除，不应推入普通 Git 仓库。
