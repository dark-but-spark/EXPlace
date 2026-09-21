# EXPlace 增量实验说明

本仓库保留原 EXPlace 的固定权重、逐步奖励路径作为 baseline。新增实验代码集中在
`src/experiments/`，通过配置显式启用；现有 `config/iccad.yaml` 和 `config/or.yaml`
不启用新逻辑。

## 改进思路

原实现使用固定的 `trade_off_coeff`，并在每放置一个 macro 后立即返回局部奖励。
这存在两个实验问题：固定权重无法判断 expert proxy 与真实 PPA 是否一致；局部奖励
可能鼓励短期收益，而不是完整布局质量。

`config/adaptive_global.yaml` 提供两项可独立消融的改进：

1. `reward_granularity: global`：中间步骤奖励为 0，在完整布局结束时返回所有步骤的
   聚合奖励；`global_reward_gamma: 1.0` 使该终局信号不会随 macro 数量指数衰减。
   设置为 `step` 即恢复原行为和原 `gamma`。
2. `reward_weight_mode: ppa_correlation`：每次 DREAMPlace/OpenTimer 评测后，计算
   `reg/hier/df/wire` 历史 cost 与下游 PPA penalty 的秩相关性。相关性更高的 proxy
   获得更高权重，并通过更新率、初始先验和最小权重限制抑制短样本震荡。

`port` 权重默认固定，因此可学习部分的总预算始终为 0.95，所有权重总和保持为 1。
当前权重会同步到 Ray 环境、Actor 的 soft mask 和模型 checkpoint。

## 运行与消融

完整改进：

```bash
python -m src.main --benchmark superblue3 --config adaptive_global --gpu 0 --seed 3
```

同一配置只测试全局奖励：

```bash
python -m src.main --benchmark superblue3 --config adaptive_global \
  --reward-weight-mode fixed
```

同一配置只测试自适应权重：

```bash
python -m src.main --benchmark superblue3 --config adaptive_global \
  --reward-granularity step
```

训练目录中的 `reward_weights.csv` 记录每次 PPA 评测的 penalty、各项权重和相关性，
TensorBoard 中对应曲线位于 `reward_weight/*`。至少积累
`reward_weight_min_observations` 次真实评测后才开始更新权重。

建议用相同 benchmark 和 seed 分别运行以下四组：`fixed+step`、`fixed+global`、
`ppa_correlation+step`、`ppa_correlation+global`，比较最佳 GP-HPWL、TNS、WNS 及
多 seed 的均值和方差。相关性只表示预测关系，不能证明因果关系，因此权重学习结果
应作为实验假设验证，而不是直接视为 PPA 最优解。
