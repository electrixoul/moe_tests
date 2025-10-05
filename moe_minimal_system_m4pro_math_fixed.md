# 超小型 MoE 序列预测实验系统（Python / MacBook Pro M4‑Pro 友好）实施手册（含数学细节补充）

> 目标：在 **单机单卡（MacBook Pro M4‑Pro）** 上，用 **PyTorch /（可选）JAX** 跑通一个**极简 MoE‑Transformer**，完成**字符级下一个 token 预测**，并配套**路由/均衡/稳定**的数学定义与实现要点，作为可靠的可复现实验基线。

---

## 0. 总览与关键配置
- **任务**：字符级 Tiny Shakespeare（极小语料、易收敛）；可选合成 **Dyck‑2**（括号语言）验证结构化分工。
- **模型**：标准 Transformer block，将 **FFN 替换为 MoE‑FFN（Top‑2 路由，E=8 专家）**；注意力保持 dense。
- **容量**：`capacity_factor≈1.0`（单机小批量基本不会拥塞）。
- **硬件**：优先 **PyTorch+MPS（Metal）**；JAX+jax‑metal 为可选路线。

---

## 1. 环境准备（M4‑Pro）
见上一版手册的第 1 节（PyTorch+MPS 验证脚本、JAX‑metal 可选安装），此处不再赘述。

---

## 2. 数据与任务
- **A. Tiny Shakespeare（字符级）**：词表 `V≈65–100`，输入序列 `x_{1:T}`，目标为 `x_{2:T+1}`，即**下一个字符预测**。
- **B. Dyck‑2（字符级）**：平衡括号语言；可控长度与嵌套度。

> 训练/验证切分：例如训练 90%，验证 10%；以固定随机种子重现。

---

## 3. 数学定义（从输入到损失）

### 3.1 记号
- 批量大小 `B`，序列长度 `T`，模型维度 `d`，专家数 `E`，Top‑k 路由 `k`（默认 `k=2`）。
- 输入 token 序列：`x ∈ {1,…,|V|}^{B×T}`。
- 词嵌入矩阵：`W_embed ∈ R^{|V|×d}`；输出投影（共享权重）`W_out = W_embed^T ∈ R^{d×|V|}`。
- 一个 MoE 层的输入隐藏状态：`H = [h_1,…,h_T] ∈ R^{B×T×d}`。

### 3.2 语言建模（LM）主损失
**条件分布**：模型给出 `p(x_{t+1} | x_{1:t})`。  
**交叉熵（teacher forcing）**：
$\mathcal{L}_{\text{CE}} = - \frac{1}{B T} \sum_{b=1}^B \sum_{t=1}^{T} \log p_\theta\!\big(x_{b,t+1}\,|\,x_{b,1:t}\big).$
具体为：先得到 logits
$Z = H^{(\text{final})} W_{\text{out}} \in \mathbb{R}^{B\times T\times |V|},
\quad p = \mathrm{softmax}(Z,\ \text{dim}=-1),$
再取目标 `x_{b,t+1}` 的对数概率求均值。

**困惑度（Perplexity）**：
$\mathrm{PPL} = \exp\!\big(\mathbb{E}[\text{NLL}]\big) = \exp\!\big(\mathcal{L}_{\text{CE}}\big).$

### 3.3 注意力与标准 FFN（简述）
单头注意力（省略 mask）
$Q = H W_Q,\quad K = H W_K,\quad V = H W_V,\quad
\mathrm{Attn}(H)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}}\Big)V.$
多头：并列拼接再线性变换。标准 FFN：
$\mathrm{FFN}(h) = W_2\,\phi(W_1 h + b_1)+b_2,\quad \phi \in \{\text{GeLU}, \text{SwiGLU}, …\}.$

### 3.4 MoE‑FFN 的路由与计算
**Router 打分与概率**（逐 token）
$s_t = h_t W_r + b_r \in \mathbb{R}^{E},\quad
p_t = \mathrm{softmax}(s_t) \in \Delta^{E-1}.$
可选 **noisy gating**：`s_t ← s_t + ε, ε∼N(0,σ^2)`（仅训练期）。

**Top‑k 选择与权重重归一化**
$\mathcal{K}_t = \mathrm{TopK}(p_t,k),\quad
g_{t,e} = \frac{p_{t,e}}{\sum_{j\in \mathcal{K}_t} p_{t,j}} \cdot \mathbf{1}\{e\in\mathcal{K}_t\}.$
其中 `g_{t,e}` 是**合并权重**（仅在被选专家内重归一化）。

**容量与分配**（占位，单卡通常不拥塞）  
令每卡 token 总数 `N=B·T`，则
$\text{capacity}=\Big\lceil \frac{N \cdot k \cdot \text{capacity\_factor}}{E}\Big\rceil.$
若某专家到达容量上限，可：`drop`（丢弃该专家上的超额 token）、`reroute`（改投第二选择）、或 `dropless`（不丢弃，用 padding 对齐）。**本最小系统在单卡上默认 dropless≈True（统计占位）**。

**Dispatch/Combine 的矩阵化视图**  
定义**路由掩码** `M ∈ {0,1}^{N×E}`，`M_{t,e}=1` 当且仅当 `e∈K_t` 且未被容量丢弃；定义**权重** `G ∈ R^{N×E}`，`G_{t,e} = g_{t,e}·M_{t,e}`。  
- 第 `e` 个专家的输入是：把所有 `M_{t,e}=1` 的 `h_t` 取出并按批拼接：  
  `X_e = [ h_t : M_{t,e}=1 ] ∈ R^{N_e×d}`。
- 专家计算：
$Y_e = f_e(X_e) = W^{(e)}_2\,\phi(W^{(e)}_1 X_e + b^{(e)}_1) + b^{(e)}_2.$
- 合并回原位：令 `y_t^{(e)}` 表示 `Y_e` 中对应该 `t` 的输出（通过记录的索引还原），则
$\hat h_t = \sum_{e=1}^{E} G_{t,e}\, y_t^{(e)}.$
实现中用 `index_select`/`gather` 打包、`scatter_add` 回收。

### 3.5 负载均衡损失（Load‑Balancing，LB）
令
$f_e = \frac{1}{N k}\sum_{t=1}^{N}\sum_{j=1}^{k}\mathbf{1}\{\text{第 }j\text{ 个 Top‑}k \text{ 为 } e\},
\quad
P_e = \frac{1}{N}\sum_{t=1}^{N} p_{t,e}.$
`f_e`：专家 `e` 实际**分到的 token 占比**；`P_e`：**平均门控概率**。  
一种常用的匹配式 LB 正则（越大代表越均衡）是 `E·∑_e f_e·P_e`，训练时采用**最小化**其相反数：
$\mathcal{L}_{\text{LB}} = -\, E \sum_{e=1}^{E} f_e\, P_e.$
> 直觉：当**分配占比**与**概率质量**在专家间一致且更均匀时，上式更大；取负最小化会**鼓励均衡**。

> 备选（可做消融）：**均匀度正则** `\sum_e (f_e - 1/E)^2`。

### 3.6 Router z‑loss（数值稳定）
对每个 token 的 logits 计算
$z_t = \mathrm{logsumexp}(s_t) = \log\!\sum_{e=1}^{E} e^{s_{t,e}},\quad
\mathcal{L}_{\text{z}} = \frac{1}{N}\sum_{t=1}^{N} z_t^2.$
`z_loss` 抑制 logits 过大，**稳定 softmax 与混合精度数值**。

### 3.7（可选）路由熵正则（探索/去确定化）
$\mathcal{L}_{\text{ent}} = -\frac{1}{N}\sum_{t=1}^{N}\sum_{e=1}^{E} p_{t,e}\log p_{t,e}.$
较小权重时可提升早期探索，减少“热专家”。

### 3.8 总损失
$\mathcal{L} = \mathcal{L}_{\text{CE}} \;+\; \lambda_{\text{lb}}\,\mathcal{L}_{\text{LB}}
\;+\; \lambda_{\text{z}}\,\mathcal{L}_{\text{z}}
\;+\; \lambda_{\text{ent}}\,\mathcal{L}_{\text{ent}} \;(\text{可选}),$
常用超参：`λ_lb≈0.02–0.1`，`λ_z≈1e-3`，`λ_ent≈(0–1e-3)`。

### 3.9 梯度流与 Top‑k
- **Top‑k 选择**使未选中的专家在该步**无主损失梯度**；选中的专家与其门控权重 `g_{t,e}` 会得到梯度。
- 通过 `p_t = softmax(s_t)` 与 `g_{t,e}` 的链式求导，**路由线性层参数 `W_r`/`b_r`** 获得梯度；`Top‑k` 的离散选择在实现里等价为对未选中通道**截断梯度**。  
- 采用 `Top‑2`（而非 `Top‑1`）通常能让更多路由与专家参数**得到梯度**，更稳、更易特化。

---

## 4. 训练与评测的数学指标

### 4.1 训练与学习率
- 优化器：AdamW；`β1=0.9, β2=0.95, weight_decay=0.1`。
- 学习率：`cosine` 退火，`warmup` 1–2k steps；最大学习率 `~3e-3`（小模型）。
- 梯度裁剪：`‖g‖₂ ≤ 1.0`。

### 4.2 评测
- **验证 NLL / PPL**：按式 (3.2)。  
- **路由统计**：`{f_e}`、`{P_e}`、路由熵 `H(p_t)`、丢弃率（若启用容量丢弃）。
- **专家分工**：按专家统计 token 类型分布；或对进入专家前的 `h_t` 做降维（t‑SNE/UMAP）并按专家着色观察簇。

---

## 5. 参考超参（能在 M4‑Pro 上稳定收敛）
- `L=4, d=384, n_head=6`；词表 `V≈65–100`（字符级）。
- `MoE：E=8, topk=2, d_hidden≈2×d, capacity_factor=1.0`。
- `B=64, T=256` 起步；bfloat16；Dropout `0–0.1`。

---

## 6. 代码映射（伪代码 → 数学式）

### 6.1 Router + Top‑k（式 3.4 / 3.5 / 3.9）
```python
# h: [B, T, d], reshape -> [N, d] where N=B*T
logits = h @ W_r + b_r            # [N, E], 相当于 s_t
probs  = softmax(logits, dim=-1)  # [N, E], 相当于 p_t

weights, indices = probs.topk(k=2, dim=-1)      # both [N, 2], K_t
weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)  # 归一化 g_{t,e}

# 构造 onehot 路由掩码 M 与权重矩阵 G（仅选中的专家位置非零）
M = one_hot(indices, num_classes=E).sum(dim=1)  # [N, E] in {0,1}
G = zeros_like(probs).scatter_add(dim=-1, index=indices, src=weights)  # [N, E]
```

### 6.2 Dispatch/Combine（式 3.4）
```python
# 收集每个专家 e 的 token 索引
token_ids_per_e = [torch.where(M[:, e] > 0)[0] for e in range(E)]
X_e = [h[idxs] for idxs in token_ids_per_e]                 # list of [N_e, d]
Y_e = [FFN_e(x) for e, x in enumerate(X_e)]                 # list of [N_e, d]

# 回收：scatter_add 到原位，并按 G 权重相乘求和（∑_e g_{t,e} y_t^{(e)}）
y = zeros_like(h)   # [N, d]
for e, idxs in enumerate(token_ids_per_e):
    y.index_add_(0, idxs, Y_e[e] * G[idxs, e].unsqueeze(-1))  # combine
y = y.view(B, T, d)
```

### 6.3 负载均衡与 z‑loss（式 3.5 / 3.6）
```python
# f_e, P_e
fe = M.float().mean(dim=0)              # [E], 约等于 (1/Nk)Σ 1{e∈K_t} (k=2 时可乘以 1/2 做严格匹配)
Pe = probs.mean(dim=0)                  # [E]

L_lb = - E * (fe * Pe).sum()            # 负号最小化 -> 鼓励均衡
z = torch.logsumexp(logits, dim=-1)     # [N]
L_z = (z ** 2).mean()
```

### 6.4 总损失与反向
```python
L = L_ce + lambda_lb * L_lb + lambda_z * L_z + lambda_ent * L_ent_optional
L.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

---

## 7. 常见问题与解决
- **路由坍缩（热专家）**：增大 `λ_lb`；启用 `noisy gating`；使用 `Top‑2`。  
- **数值不稳（bfloat16）**：启用 `z_loss`；适度降低学习率。  
- **PPL 不降**：先跑 Dense‑FFN 基线；检查 tokenizer/数据切片；降低 LR、增大步数。

---

## 8. 一键训练脚本（示意）
```bash
#!/usr/bin/env bash
set -e
source ~/.venvs/moe-mini/bin/activate

python -u moe/train.py \
  --dataset tiny_shakespeare \
  --seq_len 256 --batch_size 64 \
  --model_dim 384 --n_head 6 --n_layer 4 \
  --moe_experts 8 --topk 2 --moe_hidden_mult 2.0 \
  --lr 3e-3 --warmup_steps 2000 --wd 0.1 \
  --clip_grad 1.0 --amp bf16 \
  --lb_weight 0.05 --zloss_weight 0.001 \
  --save_dir runs/tiny_moe_top2
```

> 建议将本手册保存为仓库根目录的 `README.md`；新人照章实现，即可在 M4‑Pro 完成 MoE 最小实验闭环，并能基于上面的**数学指标**做严谨对比与改进。
