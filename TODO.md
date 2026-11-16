# EXPlace 改进 TODO

本文档只记录下一步工作，不替代历史结果。每个实验完成后，应把命令、配置、
commit/工作区状态、原始日志和结论追加到 `EXPERIMENT_ARCHIVE.md`。

## P0: 先保证可比性和可复现性

- [x] 增加 `script/capture_run_metadata.sh`，记录 Python、PyTorch/CUDA、driver、
  GPU、DREAMPlace CMake cache、git 状态和生成文件状态。
- [x] 矩阵启动时自动保存 `<RUN_ROOT>/run_metadata.txt`；任务目录仍保存
  `command.sh` 和 `resolved_config.yaml`。
- [x] 增加 `script/summarize_experiment.py`，从非零 `gp_hpwl` 行识别真实物理评估，
  分别报告 best GP、best TNS、best WNS、final 和 best-to-final。
- [x] 确认 `hcg.lp`、`vcg.lp` 是 DREAMPlace 生成物并加入 `.gitignore`；这两个文件
  在历史提交中仍被跟踪，运行产生的本地改写不得加入后续提交。
- [ ] 在正式 GPU 节点重新采集一次 cu118 基线，记录真实 GPU/driver 和 native
  `sm_XX` 架构。
- [x] 在同一 benchmark/seed 上确认 `final`、`best`、`best-to-final` 三种指标口径；
  结果归档在 `experiment_records/20260920_superblue18_4x3/`。
- [x] 增加独立 `reward_discount_gamma`（`src/agent.py:resolve_return_gamma`），把
  step 奖励折扣与 `gamma`（同时是 PLM community resolution）解耦；控制组配置为
  `config/step_gamma1.yaml`，结果归档在
  `experiment_records/20261108_superblue18_step_gamma1/`。

## P0: DREAMPlace 合法性

- [x] 修复 macro 级大错误（std cell 压 macro 的成块重叠）：
  `patches/dreamplace/macro_legalize_local_grid_repair.patch`（grid 对齐 + 重叠面积
  打分 local repair + Hannan fallback）。superblue18 已达 60/60 合法。
- [ ] **决定（2026-11-08）：sb4 的 2–9 单位薄条暂不修复。**
  - 现状：sb4 47/60（78.3%）合法，13 次失败全部是 `dreamplace_legality_check_failed`，
    形态为 std cell ↔ fixed macro 的薄条（`577×9`、`90×9`、`1940×9`），集中在 seed 7。
  - 不修的理由：(a) legality gate 会把这些 layout 判为 `invalid` 并丢弃，
    **不会污染 PPA 指标**，只降低有效样本数；(b) 已修的大错误（成块重叠）才真正影响
    legalizer 结果与评估可信度；(c) sb18 主线 60/60 不受影响。
  - 注意：不要把它简化为「DREAMPlace 对小错误容忍度高」。DREAMPlace 的
    Hannan/Abacus 擅长消解 std cell 之间的重叠，对「std cell 压 fixed macro」这类
    需要跨越固定障碍的薄条并不保证能挪开。
  - 重新打开的条件：正式对外报告 sb4 结果前，先 dump 一次失败布局，确认薄条对
    OpenTimer 数值确有可忽略影响，再决定「放宽 gate」还是「修 legalizer」。

## 已知问题: OpenTimer RC-tree 段错误（根因已定位，暂不打补丁）

2026-11-16 用 gdb + 非 LTO 重编译完成定位，完整配方见 `script/diagnostics/README.md`。

- **现象**：`exit 139`，日志最后一行 `[INFO] launch rc tree construction...`，
  无 Python traceback（崩在 C++，`try/except` 抓不到）。
- **根因**：`ot::Rct::_update_load()`（`thirdparty/OpenTimer/ot/timer/net.cpp:158`）
  递归无环守卫。`insert_segment()` 为每段插入**双向边**，而遍历只排除**直接父节点**，
  因此底层的 **3 个及以上的环** 会无限递归直到栈溢出。
  `_update_delay/_update_ldelay/_update_response` 同样写法，同样会崩。
- **触发**：某次 layout 让某个 net 的 SPEF `*RES` 出现退化/重名节点
  （`Net::_make_rct()` net.cpp:295），图闭合成环。**不是**累积内存损坏：
  同一位置重复 `update_timing()` 12 次不崩，单点回放也不崩，需要**多次不同位置**。
- **已观测失败率**：superblue18 0/24、superblue1 0/12、superblue3 2/12（约 17%）。
  失败任务会跑到中途直接丢失（无 checkpoint 续跑，重跑从 ep0 开始）。
- **决定：暂不改 OpenTimer**（visited 守卫约 20 行、对合法树零差异，但属第三方
  数值关键代码改动，产生上游同步与复现解释负担）。四个选项与代价：
  1. **保留争议 + 重跑失败任务**（当前采用，成本 0，浪费约 4h/失败任务）
  2. 评估隔离到常驻子进程（半天，需 spawn 不能 fork，CUDA 已初始化）
  3. 训练不计时、离线批量评估（1h，但 adaptive 权重学习会失效）
  4. 打补丁（1min 重编 + 10min 验证）
  → **重新评估条件**：再跑 24+ 任务、且以 superblue3 为主的矩阵前，改做选项 2。
- **取证开关**：`EXPLACE_DUMP_PRE_TIMING=<dir>` 会在每次评估前 dump 位置
  （`src/problem_instance.py`，环境变量未设时零开销）。
- ⚠️ 不要在训练进程存活时替换 `DREAMPlace/install/.../timing_cpp*.so`
  （会立刻 SIGSEGV，本排查中曾因此误判一次）。

## P0: CUDA 12 算子迁移

- [x] 增加 `script/build_dreamplace_cuda12.sh`，强制独立 build/install 目录、显式
  CUDA 架构，并自动保存命令、metadata、完整日志和第一条错误。
- [ ] 在 `DREAMPlace/build-cuda12` 中隔离编译，不覆盖当前 `DREAMPlace/install`。
- [ ] 建立失败算子清单：源文件、第一条 compiler error、CUDA API、最小修改。
- [ ] 逐个重写 CUDA 12 不兼容算子，并保留 cu118 原实现用于 A/B 对照。
- [ ] 对每个算子做 CPU/cu118/CUDA12 输出和梯度比较。
- [ ] 完成完整 DREAMPlace import、单次 placement、三次重复稳定性测试。
- [ ] 对同一输入测量 100/300/1000 iterations、legalization、OpenTimer 和显存。
- [ ] 只有通过数值和 placement-level gate 后，才将 CUDA12 标记为可比较基线。

详细记录入口：`CUDA12_MIGRATION.md`。

当前主机（2026-09-21）有 CUDA 12.2 toolkit，但 NVIDIA driver 不可用，且
`/usr/bin/python3` 的 PyTorch 是 `2.0.0+cu118`。可以做隔离编译诊断，不能完成 GPU
正确性/性能 gate。当前 preflight 在 `experiment_records/cuda12_20260921/`。

## P1: DREAMPlace 评估流水线

- [ ] 增加 short/full 两级评估：short 用于筛选，full 用于最终报告。
- [ ] 验证 100/300 iterations 与 1000 iterations 的 Spearman/Kendall 排名相关性。
- [ ] 实现独立 evaluator process；请求只传不可变 macro placement snapshot。
- [ ] 使用有界 latest-wins 队列，避免旧 placement 堵塞训练。
- [ ] 记录 evaluator version，过期结果不更新 adaptive reward weights。
- [ ] 实现 DREAMPlace standard-cell warm start，并与冷启动做单独消融。
- [ ] 评估是否可以直接更新 `dmp_placer.pos`，减少重复 PlaceDB/初始化开销。
- [ ] 将 DEF、PNG、model checkpoint 写入独立 writer process，主训练循环不等待磁盘。
- [ ] 对评估结果增加 placement hash/cache，避免相同 placement 重复跑 DREAMPlace。

## P1: RL 和数据通路效率

- [ ] profile 每个 mask generator，确认 `df/hier/wire/reg/port` 的实际耗时。
- [ ] 缓存只依赖 macro geometry 的 mask，并向量化依赖已放置 macro 的更新。
- [ ] 用 compact channel tensor 或按需重建替代完整 flattened transition。
- [ ] 让 Ray worker 按 episode 返回 rollout fragment，减少逐 step 大 observation 传输。
- [ ] 每个 PPO minibatch 只做一次 device transfer。
- [ ] 在数值对照通过后测试 CNN 分支 mixed precision。
- [ ] 单独测量 GPU utilization，区分 GPU 空转、CPU mask、PPO 和物理评估等待。

## P1: 评价指标和实验设计

- [x] 完成 `step + gamma=1` 控制组（2026-11-08）：拆开「奖励粒度」与「折扣因子」的
  混杂。结论：折扣是真实混杂项，但粒度仍有独立效果——`step+gamma=1` 峰值 TNS 最好
  （-123.88），`global+gamma=1` 终值 TNS 最好（-148.02）且最稳定（GP 回归 0.08%）。
  不得再把 H1 写成「只是折扣偏差」。详见
  `experiment_records/20261108_superblue18_step_gamma1/README.md`。
- [ ] **label fidelity（P1-1）的报告口径硬性要求**（尚未执行 P1-1 本体，仅先定口径）：
  - 必须**分电路报告** Spearman ρ 的分布与符号反转，不得只报均值。
  - 均值需用 **Fisher z 变换**后再平均，不能直接算术平均 ρ。
    （参照 PPAPlace：post-CTS 的 `ρ̄=0.38` 是由 bp_be `+0.83` 与 ethernet `−0.31`
    平均出来的，均值掩盖了符号反转；反之 post-GRT 的 `ρ̄=0.86` 在 10/10 电路上都
    `ρ≥0.78` 且无反转，均值可信。）
  - 决策门仍为 ρ>0.7 继续 / 0.4–0.7 降级 / <0.4 停止权重学习，但必须按电路分别判。
- [x] 增加 seed `17/23/29` 与 `superblue1`、`superblue3`（2026-11-09，P1-2）：
  24 任务 22/24 成功（2 个 OpenTimer SIGSEGV），6 seeds × 2 电路 × fixed/adaptive_global。
  **结果：sb18 的 adaptive 优势没有复现** —— post-adaptation 窗口内 TNS 5胜/7负
  （p=0.77）、best GP-HPWL 5胜/7负（p=0.77）；唯一可复现的正向信号是 WNS
  （10胜/2负，p=0.039），且 final GP-HPWL 小幅改善 1–3%。
  详见 `experiment_records/20261109_p12_multidesign/README.md`。
- [ ] **`best_*` 指标的口径硬性要求（2026-11-09）**：对 `adaptive_*` 臂，任何 `best_*`
  比较必须限定在 **episode > 首次权重更新 episode**（默认 400，应从
  `reward_weights.csv` 推导）。因为更新前的评估在两臂之间逐位相同，会把配对差压成 0
  并制造假「平局」。实现见
  `experiment_records/20261109_p12_multidesign/analyze_post_adaptation.py`。
  `script/summarize_experiment.py` 的全程 `best_*` 仅适用于 fixed 臂或只看 `final`。
- [ ] **修复 OpenTimer RC-tree 段错误**：`superblue3` 上 2/24 任务在
  `launch rc tree construction...` 处 SIGSEGV（exit 139），合法性检查已通过，
  发生在时序分析阶段，非 legality 失败。目前导致任务丢失（只能跑到 ep800）。
  需要：定位是否与 placement 边界/电源网有关，并加重试或跳过该评估的容错。
- [ ] 补充 rWL、rOverflow、congestion、DRC；条件允许时补 Power/Area/NVP。
- [ ] 明确 post-placement OpenTimer、post-GRT、post-DRT 的评估阶段和量纲。
- [ ] 使用固定 checkpoint/validation 选择，减少周期物理评估后取 best 的选择偏差。
- [ ] 增加 `port` 可学习对照，确认 `.05` 不漂移是约束还是数据结果。
- [ ] 报告每个 seed 的绝对值、均值、方差和 best-to-final，而不只报百分比。
- [ ] 将 `fixed_global` 保留为反例，验证 global reward 单独是否足够。

## P2: PPA surrogate 方向

- [ ] 收集 placement、proxy costs、GP-HPWL、TNS、WNS 的统一数据集。
- [ ] 先训练 post-placement surrogate，用于 checkpoint ranking，不直接改变 PPO reward。
- [ ] 验证 surrogate 排名与真实 DREAMPlace/OpenTimer 的相关性。
- [ ] 再考虑加入 Post-GRT 标签和 PPAPlace 风格 dual-stream predictor。
- [ ] 仅在 surrogate 稳定后，将其用于 adaptive weight 或 terminal reward。

## 暂不做

- [ ] 不在同一个 `ProblemInstance`/`PlaceDB` 对象上并发调用 DREAMPlace。
- [ ] 不把 cu118 PyTorch 与未经验证的 CUDA12 extension 当作正式性能基线。
- [ ] 不用 GP-HPWL 单独宣称 PPA 改善。
- [ ] 不把单 benchmark、3 seed 的结果写成统计显著或普适结论。
- [ ] 不把本项目绝对 TNS/WNS 与论文的 32.53%/7.74% 直接比较：论文口径是相对
  runner-up 的提升，本项目目前只报绝对指标，且未复现论文的 runner-up 基线。
