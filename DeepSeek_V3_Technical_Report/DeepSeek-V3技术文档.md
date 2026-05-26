# DeepSeek-V3技术文档

> DeepSeek-V3 Technical Report
>
> 2026.5.25-

## 解决的问题

* strong model performance and economical costs.

* 在如此大规模的模型上把 FP8 训练做稳定，业界之前很少有人成功过。

------

## Intro

DeepSeek-V3, a large Mixture-of-Experts (MoE) model with 671B parameters, of which 37B are activated for each token.

------

### 架构

模型架构采用以下两种在V2就被验证有效的架构：

* Multi-head Latent Attention (MLA)

* DeepSeekMoE

除了以上两种基础架构以外，V3还引入了两种策略来增强模型能力：

*  auxiliary-loss-free strategy

  > 负载平衡
  > “minimizing the adverse impact on model performance that arises from the effort to encourage load balancing”

*  multi-token prediction training objective

  > 提升评估基准的整体表现
  >
  > “enhance the overall performance on evaluation benchmarks”

------

### 训练精度

we support the FP8 mixed precision training and implement comprehensive optimizations for the training framework.

第一次将==FP8的混合精度训练==应用到了 an extremely large-scale model

#### 优点

##### 1. 计算速度更快

- H100 GPU 的 FP8 Tensor Core 算力是 BF16 的 **2倍**
- 矩阵乘法（GEMM）吞吐量大幅提升
- 训练速度整体可提升 **30%~50%**

##### 2. 显存占用减半

- 相比 FP16/BF16，显存节省 **50%**
- 可以训练**更大的模型**，或使用**更大的 batch size**

##### 3. 通信带宽降低

- 分布式训练中，梯度通信量减少
- 对多机多卡训练尤其友好

#### 缺点

##### 1. 数值表示范围极窄

FP8 只有8位，能表示的数值范围远小于FP16：

| 格式           | 指数位 | 尾数位 | 可表示范围             |
| -------------- | ------ | ------ | ---------------------- |
| FP32           | 8位    | 23位   | ~1.2×10⁻³⁸ 到 3.4×10³⁸ |
| BF16           | 8位    | 7位    | ~1.2×10⁻³⁸ 到 3.4×10³⁸ |
| FP16           | 5位    | 10位   | ~6×10⁻⁸ 到 65504       |
| **FP8 (E4M3)** | 4位    | 3位    | **~1.95×10⁻³ 到 448**  |

##### 2. 精度损失 → 训练不稳定

- 尾数位只有3位，舍入误差大
- 梯度容易**上溢（overflow）或下溢（underflow）**
- 严重时导致训练**直接发散**

##### 3. 需要精细的 Loss Scaling

- 必须动态调整缩放因子，把数值"搬"到FP8能表示的范围内
- 实现复杂，调参困难

##### 4. 硬件支持有限

- 只有 **NVIDIA H100/H800** 及以后的GPU才有原生FP8支持
- A100 及更早的卡**不支持**，无法受益

##### 5. 并非所有算子都适合

- Softmax、LayerNorm、激活函数等对精度敏感的算子**不能用FP8**
- 需要精心设计哪些层用FP8，哪些层回退到BF16

------

### 训练架构

DualPipe algorithm for efficient pipeline parallelism（用于高效管道并行的 DualPipe 算法）

> 通过计算通信重叠，减少管道气泡并隐藏训练期间的大部分通信（“fewer pipeline bubbles and hides most of the communication during training through computation-communication overlap”）
>
> overlap就是类似于CPU的并行流水线，如下所示：
>
> ```
> 时间轴 ──────────────────────────────────►
> 
> 计算:  [  计算A  ][  计算B  ][  计算C  ]
> 通信:            [  通信A  ][  通信B  ]
>                      ↑
>                 两者同时进行
> ```
>
> 这就是所谓的computation-communication overlap

------

#### 高效的跨节点全对全通信内核（efficient cross-node all-to-all communication kernels）

可以完全使用InfiniBand和NVLink的全部带宽，

------

#### 精细优化内存占用

使训练DeepSeek-V3时不需要使用昂贵的张量并行处理。

------

#### Pre-training & Post-training

```
  海量原始文本
      ↓
  Pre-training    ← 学语言能力（知识、逻辑、语法）
      ↓
  Base Model
      ↓
  Post-training   ← 学行为方式（听指令、对话、安全）
      ↓
  最终产品（如 DeepSeek V3 / ChatGPT）
```

在Pre-training阶段，DeepDeek-V3采用了两个阶段的上下文扩展（上下文拓展一般在完成主预训练之后）。第一阶段，拓展到32K；第二阶段，拓展到128K。

在Base Model基础上，进行Post-training阶段。该阶段包含Supervised Fine-Tuning (SFT，监督微调）和Reinforcement Learning（RL，强化学习）。此外，还蒸馏了DeepSeek-R1系列模型，同时精细保持了模型正确率和生成长度之间的平衡。

------

## 技术列表

- [ ] Multi-head Latent Attention(MLA，多头潜在注意力) 
- [ ] DeepSeekMoE architectures 
- [ ] auxiliary-loss-free strategy(辅助无损失策略)
- [ ] multi-token prediction training objective(多标记预测训练目标)

-----

## 核心创新点

