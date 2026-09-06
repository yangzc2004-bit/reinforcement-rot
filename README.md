# Reinforcement Rot

> 让多智能体系统越用越好的机制，和让它无报错空转到死的机制，是同一个机制。

多智能体 LLM 系统（MAS）的共享经验记忆是一个**正反馈回路**：被使用的经验变强，变强的经验更被使用。本研究计划刻画一个静默失效模式——**Reinforcement Rot**：不需要任何 agent 出错，一次微小的"绕路"经验被正反馈传播放大，整个群体锁死进无报错的死循环（蚂蚁死亡螺旋的数字对应物）。

**核心论点**：当前所有主流 MAS 记忆设计只有**强化**、没有**蒸发**——而生物学里正是蒸发把正反馈从"随机锁定器"变成"最优性搜索引擎"。

## 三层架构

| 层 | 内容 | 状态 |
|---|---|---|
| L1 理论 | 瓮模型 + 带环捕获项的平均场 ODE（`theory/`） | ✅ 结构完成（2026-08 λ 语义已修正，见下） |
| L2 仿真 | 字面信息素双桥+环世界，规则 agent（`sim/`） | ✅ 相图已复验 |
| L3 LLM | LLM 智能体走 15×15 受控拓扑迷宫，共享语言记忆（`llm/`） | ✅ 引擎就绪（B1/B2/β_eff 可跑，待填 API） |

## 目前已确立的结果（L1+L2）

1. **随机锁定区**：无蒸发（λ=1）时系统结局由早期随机涨落决定，跨种子方差极大；λ ≤ 0.85 时方差塌缩（预言 T1 ✅）。
2. **磨盘带**：死亡螺旋不住在极端，而住在中间地带 **λ ∈ [0.93, 0.97]**——下边界是"磨盘能否存活"（寿命问题），上边界是"磨盘来不来得及成形"（干道锁定与环捕获的赛跑）。
3. **微小扰动最危险**：磨盘带集中在 δ=1–2（几乎和最优一样好的绕路），δ=6 以上基本消失（预言 T2 ✅）。
4. **超线性响应是磨盘的必要条件**：跟随响应指数 β=1.5 时磨盘绝种（寿命恒 ~100 tick），β≥2 才可能出现。L3 不预设"top-k = β→∞"——每个模型的有效跟随锐度 β_eff 由 `llm/calib_beta.py` 实测。
5. **磨盘寿命曲线** τ_mill(λ, β)：β≥2 时寿命随 λ 增长；临界质量标度 n_c ∝ κ^(1/β)。

## λ 的单位约定（2026-08 修正）

**λ 一律是"逐 tick 信息素保留率"**。仿真端每 10 tick 执行 `scent *= λ**10`；
理论端 κ = −ln(λ)（逐 tick）。2026-08 之前 meanfield.py 误用 κ = −ln(λ)/10，
使理论蒸发比仿真弱 10 倍——定性结论（磨盘带、β 阈值）修正后全部存活，
绝对寿命数值缩小；L2 全部原始数据不受影响（仿真未动）。

## 目录结构

```
theory/meanfield.py        平均场 ODE：稳定性探针、磨盘寿命曲线（κ = −ln λ）
sim/bridge_world.py        L2 仿真核心：双桥+环图世界、信息素规则 agent
experiments/sweep_phase.py     相图扫描（7λ × 5δ × 12 seeds = 420 runs, T=4000）
experiments/verify_robust.py   复验：加密相图 + 实现参数扰动
experiments/validate_mazes.py  L3 迷宫拓扑不变量 ×1000 seeds 验证
docs/research-plan.md      研究计划（已实现 vs 目标设计，分层标注）
llm/                       L3 LLM 层：受控迷宫 / 共享记忆 / 五判据 / B1·B2·β_eff
tests/                     单元测试 + Mock 端到端（CI 门禁）
results/                   L2 数据（gzip+base64 分块，decode.py 解码，SHA 校验）
```

## 复现

```bash
pip install -r requirements.txt
python -m unittest discover tests     # 单元测试 + Mock 端到端
python experiments/sweep_phase.py     # 相图扫描（420 runs）
python experiments/verify_robust.py   # 复验扫描
python experiments/validate_mazes.py  # L3 迷宫不变量（1000 seeds）
cd llm && cp config.example.yaml config.yaml  # 填入模型 API 后：
cd .. && python -m llm.run_b1         # B1 能力门槛（80–90%）
python -m llm.calib_beta              # 模型 β_eff 标定（模型卡片）
python -m llm.run_b2                  # B2 存在性实验（可断点续跑）
```

现象命名：Reinforcement Rot；机制命名：reinforcement without evaporation。
