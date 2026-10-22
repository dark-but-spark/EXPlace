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

- [ ] 增加 seed `17/23/29`。
- [ ] 增加 `superblue1`、`superblue3`，至少保留 `superblue18` 对照。
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
