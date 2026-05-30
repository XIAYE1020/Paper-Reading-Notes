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

### 模型架构改进

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

### 训练架构改进

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

## 架构

![image-20260526142037535](./DeepSeek-V3技术文档.assets/image-20260526142037535.png)

### Transformer Block（整体框架）

```
输入
 ↓
RMSNorm        ← 归一化
 ↓
Attention      ← 注意力机制（展开就是右下角的MLA）
 ↓
⊕              ← 残差连接（把输入直接加回来）
 ↓
RMSNorm        ← 再次归一化
 ↓
Feed-Forward   ← 前馈网络（展开就是右上角的MoE）
 ↓
⊕              ← 残差连接
 ↓
输出

以上这个Block重复 L 次
```

**本质**：这就是标准Transformer结构，DeepSeek的创新在于把里面两个核心组件换掉了。

------

## 任务列表

- [x] RMSNorm（均方根归一化）
- [x] 训练精度

- [x] 词嵌入

- [x] Multi-head Latent Attention(MLA，多头潜在注意力) 
- [x] DeepSeekMoE architectures 
- [ ] Auxiliary-loss-free strategy(辅助无损失策略)
- [ ] Multi-token prediction training objective(多标记预测训练目标)

-----

## 笔记

### 一、RMSNorm

#### 先从"为什么需要归一化"说起

想象你在训练模型，数据在网络里一层一层传递：

```
第1层输出：  [0.1,  0.2,  0.3]   ← 数值很小
第2层输出：  [1.2,  3.5,  2.1]   ← 数值变大了
第3层输出：  [45,   123,  67]    ← 数值更大了
第4层输出：  [9999, 23456, ...]  ← 爆炸了！
```

数值越来越大，会导致：

- 梯度爆炸 → 训练崩溃
- 数值不稳定 → 结果乱掉

#### 归一化就是"把数值拉回正常范围"

用一个生活比喻：

> 班级考试，有人考了985分（满分100），有人考了0.3分，分数尺度完全不一样，老师没法比较。
>
> 归一化就是**把所有分数统一换算到同一个尺度**，比如0到1之间。

```
归一化前：  [0.1,  500,  23,  9999]
归一化后：  [0.11, 0.43, 0.28, 1.0]  ← 都在合理范围内
```

#### RMSNorm 具体怎么做？

RMS 是 **Root Mean Square（均方根）** 的缩写。

它的做法非常简单：

```
第一步：计算所有数值的"平均大小"（RMS）
第二步：把每个数都除以这个"平均大小"

例子：
输入：[3, 4]
RMS = √((3²+4²)/2) = √(25/2) ≈ 3.54

归一化后：
[3/3.54, 4/3.54] = [0.85, 1.13]

✓ 数值被控制住了
```

#### 为什么叫 RMS"Norm"，不用其他归一化？

常见归一化对比：

| 方法        | 做法            | 特点                 |
| ----------- | --------------- | -------------------- |
| LayerNorm   | 减均值 ÷ 标准差 | 经典，计算略复杂     |
| **RMSNorm** | **只 ÷ 均方根** | **更简单，速度更快** |
| BatchNorm   | 按batch归一化   | 不适合语言模型       |

RMSNorm 省略了"减均值"这一步，**计算量更少**，但效果几乎一样好，所以现代大模型（LLaMA、DeepSeek等）都用它。

#### 在 Transformer 里它放在哪？

```
输入数据
   ↓
RMSNorm   ← 先归一化，让数值稳定
   ↓
Attention  ← 再做复杂计算
   ↓
⊕ 残差连接
   ↓
RMSNorm   ← 再次归一化
   ↓
FFN/MoE
   ↓
⊕ 残差连接
```

每次做复杂运算**之前**先归一化，就像每次做精密实验前先**校准仪器**，保证输入数据始终在合理范围内。

#### 一句话总结

> RMSNorm 就是**防止数值在网络里越传越大、越来越乱**的"稳定器"，让每一层的输入都保持在合理范围，训练才能稳定进行。

------

### 二、训练精度（FP8）

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

### 三、词嵌入

自然语言处理（NLP）

对于以下三句话，使用不同的词袋法：

<img src="./DeepSeek-V3技术文档.assets/image-20260528102818685.png" alt="image-20260528102818685" style="zoom:50%;" />

得到不同的表格：

<img src="./DeepSeek-V3技术文档.assets/image-20260528102728985.png" alt="image-20260528102728985" style="zoom:50%;" />

对于日常文本中，词频高的词不一定是重要的，因此出现了词袋法的升级版:==TF-IDF==法（词频*稀有性）

<img src="./DeepSeek-V3技术文档.assets/image-20260528103646211.png" alt="image-20260528103646211" style="zoom:50%;" />

词频：单个词的重复数/总词数（去重）

稀有：出现该词的文档/总文档数



==词嵌入（Word Embedding）==：

对于一部电影的评价，好和棒是一个意思，但是却会被分成两个词，在本质上是因为无法理解词义

google提出Word2Vec，如果“好”和“棒”周围经常有相同词出现，那他们就是类似的。

训练过程如下

``` 
one-hot词编码
    ↓
Embedding矩阵（核心）
    ↓
  输出层
    ↓
预测上下文词
```

One-hot词编码*Embedding矩阵，就类似于查表，查到当前词在Embedding矩阵的各维度信息

通过神经网络训练，最终得到词表矩阵，记录每个词的维度信息

*使用余弦相似度计算向量相关性*，这样判断词之间的相关性。



但是上述方法==没有考虑词语的顺序问题==，如：我爱你/你爱我，两句话意思完全不同。

此外也没有考虑一词多义问题，如：apple和Apple，一个水果，一个手机，使用一行就包含了多种意思，这是不可行的。





### 四、Multi-head Latent Attention (MLA)

在模型推理过程中，Key、Value矩阵需要保存下来，需要占用大量的显存空间。DeepSeekV3采用压缩K、V矩阵的方式，在生成过程中，仅需cache蓝色标记框即可。

![image-20260530105338864](./DeepSeek-V3技术文档.assets/image-20260530105338864.png)

对于Query矩阵，DeepSeek也采用压缩的方式

![image-20260530104757879](./DeepSeek-V3技术文档.assets/image-20260530104757879.png)

------

### 五、DeepSeekMoE architectures

#### 主要公式

DeepSeekMoE的最终输出公式如下：
$$
h_t' = u_t
+ \sum_{i=1}^{N_s} FFN_i^{(s)}(u_t)
+ \sum_{i=1}^{N_r} g_{i,t} FFN_i^{(r)}(u_t)
$$

| 符号          | 含义                                |
| ------------- | ----------------------------------- |
| $u_t$         | 第 t 个 token 输入 FFN 前的隐藏状态 |
| $h_t'$        | FFN层最终输出                       |
| $N_s$         | Shared Expert（共享专家）数量       |
| $N_r$         | Routed Expert（路由专家）数量       |
| $FFN_i^{(s)}$ | 第 i 个共享专家                     |
| $FFN_i^{(r)}$ | 第 i 个路由专家                     |
| $g_{i,t}$     | token 对专家 i 的门控权重           |

第一项 $u_t$:残差连接

第二项 $\sum_{i=1}^{N_s} FFN_i^{(s)}(u_t)$:共享专家输出

第三项$\sum_{i=1}^{N_r} g_{i,t} FFN_i^{(r)}(u_t)$:路由专家部分

------

DeepSeekMoE与传统MoE的差别在于：

传统MoE采用的是

```
    token
      ↓
    Router
      ↓
只选择几个专家
```

DeepSeekMoE采用的是

```
     token
       ↓
 ┌─────────────┐
 │ Shared专家   │ ← 永远执行
 └─────────────┘
       
 ┌─────────────┐
 │ Routed专家   │ ← Top-K选择
 └─────────────┘
```

-------

#### $g_{i,t}$是如何得到的？

##### 1

首先计算匹配度$s_{i,t}$，是Router的核心：
$$
s_{i,t}=Sigmoid(u_t^T e_i)
$$
变量详解：

| 符号       | 含义                                |
| ---------- | ----------------------------------- |
| $s_{i,t}$  | token与专家i的匹配度                |
| $u_t$      | token表示                           |
| $e_i$      | 专家i的中心向量（Expert Embedding） |
| $u_t^Te_i$ | 点积相似度                          |

> 在DeepSeekV2中，采用的是==softmax==函数来计算$s_{i,t}$，而softmax函数会让专家竞争，一个专家的得分高了，其余的专家得分就低了。
>
> 因此，为了评选出Top-K的专家，DeepSeekV3采用了==Sigmoid==来为专家评分，每个专家都有自己独立的评分，更加客观。找到多个适合的专家。

##### 2

Top-K专家的选择，使用如下公式：
$$
g'_{i,t}=
\begin{cases}
s_{i,t},&s_{i,t}\in\operatorname{Topk}(\{s_{j,t}\},K_r)\\
0,&\text{otherwise}
\end{cases}
$$
简而言之：当前专家的匹配度是前K个时，保留；反之，则将匹配度直接清零。

##### 3

将门控权重归一化，公式如下：
$$
g_{i,t}
=
\frac{g'_{i,t}}
{\sum_{j=1}^{N_r} g'_{j,t}}
$$
归一化的原因：不同的token得到的$g'_{i,t}$总和不一样，因此要归一化。



