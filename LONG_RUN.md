# 长时间实验矩阵

`script/run_experiment_matrix.sh` 使用相同的 `adaptive_global` 训练配置运行四组消融：

| variant | 权重 | 奖励粒度 |
| --- | --- | --- |
| `fixed_step` | 固定 | 逐步奖励，原始 baseline |
| `fixed_global` | 固定 | 全局终局奖励 |
| `adaptive_step` | PPA 相关性自适应 | 逐步奖励 |
| `adaptive_global` | PPA 相关性自适应 | 全局终局奖励 |

默认运行 `superblue18`、seed 3、GPU 0，共四个 1500-episode 任务。单张 GPU
上的任务串行执行；提供多个 GPU 时，每张 GPU 启动一个 worker 并行消费任务。

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
   SEEDS='3 7 11' GPU_IDS='0 1' \
   bash script/run_experiment_matrix.sh"
```

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
`ALLOW_CPU` 和 `REQUIRE_PREPROCESSED`。

正式长跑默认要求 PyTorch 能看到所指定的 GPU，并要求每个 benchmark 已存在
`preprocessed_data/<benchmark>.pt`。只有明确希望 CPU 长跑时才设置 `ALLOW_CPU=1`；
只有准备让程序现场生成预处理数据时才设置 `REQUIRE_PREPROCESSED=0`。

脚本会自动选择能够导入 `torch`、`ray` 和 `yaml` 的 Python；当前 Conda 环境缺少
这些依赖时会回退到 `/usr/bin/python3`。也可以通过
`PYTHON_BIN=/absolute/path/to/python` 显式指定环境。
