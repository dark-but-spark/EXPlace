# 长时间实验矩阵

`script/run_experiment_matrix.sh` 使用相同的 `adaptive_global` 训练配置运行四组消融：

| variant | 权重 | 奖励粒度 |
| --- | --- | --- |
| `fixed_step` | 固定 | 逐步奖励，原始 baseline |
| `fixed_global` | 固定 | 全局终局奖励 |
| `adaptive_step` | PPA 相关性自适应 | 逐步奖励 |
| `adaptive_global` | PPA 相关性自适应 | 全局终局奖励 |

Reward 矩阵默认 `EXECUTION_MODE=legacy`，关闭 compact transition、worker rollout、
mask/device-copy 优化、AMP 和异步 DREAMPlace，按原有执行路径比较四种 reward 策略。
如果只希望保留性能优化、但仍使用同步 DREAMPlace，设置
`EXECUTION_MODE=optimized`（默认 `ASYNC_EVALUATION=0`）。只有专门测试异步评估时才在
optimized 模式显式设置 `ASYNC_EVALUATION=1`；该模式仍有 evaluator 卡住的风险。

同步和异步的 periodic 物理评估现在共用自适应权重更新逻辑：用被评估轨迹的
proxy costs 与对应 GP-HPWL/TNS/WNS 更新 learner，再同步到 agent、测试环境和
所有 rollout worker，并写入 `reward_weights.csv` / TensorBoard。
`fixed` 模式不进行权重更新。

物理评估现在还执行最终 DREAMPlace 合法性门禁。边界、site/row 对齐、宏/单元
重叠或 fence-region 检查失败的结果会写入 `physical_evaluations.csv`，并标记
`valid=0` 及 `failure_reason`；这类结果不会更新 best checkpoint，也不会进入
adaptive learner。旧的、没有 `valid` 列的 CSV 只能按历史兼容路径读取，不能据此
证明布局合法。宏合法化的最后 site/row 对齐也会尝试有限的邻近物理网格修复，找
不到合法位置时仍拒绝该评估，不会把非法坐标伪装成成功。

注意：`long_runs/global_multidesign_legacy_20261008_111740` 的同步路径漏掉了
learner 观测，两个 adaptive 臂实际保持初始固定权重，不能用于评价自适应效果。
保留该目录作为历史记录；修复后的 adaptive 臂应使用新的 `RUN_ROOT` 重跑，
否则 `RESUME=1` 会跳过旧的 `SUCCESS`。原有 fixed 臂可继续用于相同设置的对照。

短测须覆盖足够的物理评估：默认 `reward_weight_min_observations=4`、
`evaluation_interval=100`，至少 episode 400 才触发首次更新；只跑 20 episodes
不会验证自适应。若缩短测试，应将评估间隔调整为 rollout batch size 的整数倍，
覆盖至少四次评估及更新后的一个训练 batch。检查 CSV 中是否出现 `updated=1`，
并确认 learnable 权重改变，固定的 `port` 权重仍为 0.05。

默认运行 `superblue18`、seed 3、四个 1500-episode 任务。单张 GPU 上的任务串行
执行；提供多个 GPU 时，每张 GPU 启动一个 worker 并行消费任务。后续 L40S 验证请
显式使用 `GPU_IDS='3 5'`，不要依赖 launcher 默认 GPU 或 `auto` 型号选择。

## 启动

先检查任务矩阵，不开始训练：

```bash
DRY_RUN=1 bash script/run_experiment_matrix.sh
```

在 tmux 中启动默认长跑：

```bash
tmux new-session -d -s explace-long \
  "cd /data0/cjy_cxsj/EXPlace-test/EXPlace && bash script/run_experiment_matrix.sh"
```

多 benchmark、多 seed、两张 GPU：

```bash
tmux new-session -d -s explace-long \
  "cd /data0/cjy_cxsj/EXPlace-test/EXPlace && \
   BENCHMARKS='superblue18 superblue1 superblue3' \
   SEEDS='3 7 11' GPU_IDS='3 5' \
   bash script/run_experiment_matrix.sh"
```

本机后续多电路验证优先使用物理 GPU 3 和 5（L40S）。启动前用
`nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits`
确认索引与型号；使用 `GPU_IDS='3 5'`，不要用会选择任意空闲卡的 `GPU_IDS=auto`。

查看 tmux 输出：

```bash
tmux attach -t explace-long
```

脚本启动时会打印本次 `RUN_ROOT`。另一个终端可查看总进度：

```bash
tail -f long_runs/<timestamp>/master.log
column -s, -t long_runs/<timestamp>/summary.csv
```

每个任务目录包含 `command.sh`、`stdout.log`、`SUCCESS` 或 `FAILED`，模型与指标位于
该目录的 `output/rl_logs/<benchmark>/<timestamp>/`。`summary.csv` 汇总退出码和耗时。

## 恢复与筛选

中断后使用相同的 `RUN_ROOT` 重启；默认 `RESUME=1`，已有 `SUCCESS` 的任务会跳过：

```bash
RUN_ROOT=long_runs/<timestamp> bash script/run_experiment_matrix.sh
```

只运行部分消融：

```bash
VARIANTS='fixed_step adaptive_global' bash script/run_experiment_matrix.sh
```

默认单任务失败后继续执行其他任务。设置 `FAIL_FAST=1` 可让对应 GPU worker 在首次
失败后停止。完整可配置环境变量为 `BENCHMARKS`、`SEEDS`、`GPU_IDS`、`VARIANTS`、
`CONFIG`、`RUN_ROOT`、`PYTHON_BIN`、`RESUME`、`FAIL_FAST`、`DRY_RUN`、
`ALLOW_CPU`、`REQUIRE_PREPROCESSED`、`TASK_ORDER`、`EXECUTION_MODE` 和
`ASYNC_EVALUATION`。

正式长跑默认要求 PyTorch 能看到所指定的 GPU，并要求每个 benchmark 已存在
`preprocessed_data/<benchmark>.pt`。只有明确希望 CPU 长跑时才设置 `ALLOW_CPU=1`；
只有准备让程序现场生成预处理数据时才设置 `REQUIRE_PREPROCESSED=0`。

脚本会自动选择能够导入 `torch`、`ray` 和 `yaml` 的 Python；当前 Conda 环境缺少
这些依赖时会回退到 `/usr/bin/python3`。也可以通过
`PYTHON_BIN=/absolute/path/to/python` 显式指定环境。
