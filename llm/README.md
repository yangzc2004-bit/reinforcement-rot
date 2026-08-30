# L3: LLM 智能体层（待建）

L1（理论）与 L2（仿真）已闭环。本层是论文的 LLM 实验层，按计划从 **B1 守门实验**开始。

## 待建组件

- `env/`：网格迷宫生成器——构造式生成（先埋最短路，再埋 δ 可控绕路，再加闭环走廊）
- `agent/`：solver 循环（局部观察 → top-k 检索注入 → 逐步决策）+ reviewer（经验蒸馏与巩固）
- `memory/`：共享经验池（强度字段、蒸发钩子 ×λ、写入规则开关：LLM 评判全量 / 仅成功 / 仅高效成功）
- `audit/`：度量面板——cycle_rate / excess_steps / mill_rate / escape_time / death_rate + 写侧审计

## 实验矩阵（详见 docs/research-plan.md）

B1 能力守门 → B2 现象存在性 → B3 扰动注入剂量-响应（全文心脏）→ B4 蒸发剂量-响应 →
B5 写入规则消融 → B6 群体规模 → B7 长视野 → C1 跨模型 → C2 真实框架审计（LangGraph）→ C3 待定
