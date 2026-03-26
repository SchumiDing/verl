# verl_wanjuan 训练框架功能扩展说明

本路径下的修改主要围绕化学反应预测（逆合成与前向验证）以及复杂逻辑任务的强化学习训练展开。以下是三大核心功能的修改逻辑与设计思路。

---

## 1. 前向验证强化学习 (Forward RL)

### 修改背景
在逆合成分析（Retrosynthesis）中，模型需要给出“原料 A + 原料 B -> 目标产物 P”的预测。传统的强化学习仅对比预测的原料与标准答案是否一致，但这忽略了化学反应的“多解性”：不同的原料组合可能都能生成同一个产物。

### 修改逻辑
- **引入外部验证模型**：在 `RewardManager` 中通过 `VLLMBeamSearchManager` 集成了一个预训练的前向预测模型（Forward Model）。
- **异步推理流程**：
    1. 策略模型（Policy）生成原料 SMILES。
    2. 奖励管理器提取这些 SMILES，通过 `run_batch_forward` 异步发送给前向模型。
    3. 前向模型预测产物，奖励管理器通过 RDKit 计算预测产物与原始目标产物的 Tanimoto 相似度。
- **资源共享**：利用 Ray 的 `placement_group` 机制，使奖励节点能高效访问共享的 GPU 资源。

### 设计思路
将“生成-验证”闭环引入 RL。即便 Policy 生成的原料不在标准答案中，只要前向验证能证明其有效，模型也能获得正向反馈。这极大提升了模型探索新化学路径的能力。

---

## 2. 化学领域专属训练 (Chemical Training)

### 修改背景
化学分子以 SMILES 字符串表示，具有严格的拓扑结构要求。简单的字符串匹配（String Match）无法衡量分子间的本质相似度。

### 修改逻辑
- **集成 RDKit 工具链**：在 `RewardManager` 中引入 `rdkit.Chem`。
- **多维度评估指标**：
    - **合法性奖励**：能被 RDKit 解析的 SMILES 给予基础分，非法则重罚（-2.0）。
    - **规范化匹配**：强制转换为 Canonical SMILES 后进行对比，解决手性、电荷等表示差异。
    - **Tanimoto 相似度**：利用 Morgan Fingerprint 计算分子指纹相似度。当相似度超过阈值（如 0.85）时，给予梯度式的软奖励，引导模型向正确结构靠近。

### 设计思路
将离散的字符串反馈转化为更连续的化学空间相似度反馈，解决了强化学习在处理极稀疏奖励（Sparse Reward）任务时的收敛难题。

---

## 3. 多路径蒸馏训练 (Multiroute Training)

### 修改背景
一个化学产物往往有多种合成路线。如果训练集只有一条路径，模型容易陷入过拟合。我们需要一种机制，让模型在一个 Batch 中学会覆盖所有已知路径。

### 修改逻辑
- **Rollout 聚合处理**：在 `run_single` 中，一次性接收同一 Query 的全部 $N$ 个 Rollout 样本（`rollout_n`）。
- **覆盖率统计**：统计这 $N$ 个样本中，分别命中了标准答案中的哪几条路径。
- **动态替换 (Dynamic Substitution)**：
    - 如果标准答案有 3 条路径，但模型只生成了 1 条。
    - 逻辑会自动挑选出生成失败（错误）的 Rollout，将其内容强制替换为缺失的标准路径（Ground Truth）。
    - 这种替换是在推理后、计算损失前发生的，相当于一种“在线教师演示”。
- **梯度掩码 (Gradient Masking)**：通过 `drop_mask` 机制，可以选择性地忽略某些重复命中的样本，确保模型在每个 Step 都向着“未掌握”的路径优化。

### 设计思路
将 RL 与蒸馏（Distillation）结合。通过在 Rollout 层面进行动态补全，强制模型在同一个训练步内感知并学习该问题的完整解空间，从而实现从“单路径记忆”到“多路径推理”的跨越。

---

## 关键代码位置
- **Forward RL**: `verl/experimental/reward_loop/reward_manager/forward_rdkit_cot.py`
- **Multiroute**: `verl/experimental/reward_loop/reward_manager/multiroute_distill_cot.py`
- **核心逻辑接入**: `verl/trainer/main_ppo.py` 中对 `reward_manager` 的调用流。
