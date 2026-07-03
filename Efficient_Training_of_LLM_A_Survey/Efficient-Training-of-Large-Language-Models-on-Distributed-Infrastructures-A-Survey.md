# Efficient Training of Large Language Models on Distributed Infrastructures: A Survey

## 0. 论文基本信息

- 标题: Efficient Training of Large Language Models on Distributed Infrastructures: A Survey
- 类型: Survey
- 主题: 大语言模型分布式训练系统与基础设施
- 核心关键词: LLM Training, Distributed Training, Scalability, Efficiency, Reliability, Parallelism, Fault Tolerance
- 阅读日期:
- 阅读进度:

## 1. 一句话总结

> 这篇论文系统梳理了大语言模型在分布式基础设施上高效训练所面临的核心问题，以及围绕基础设施、并行策略、计算优化、内存优化、通信优化和容错机制的主要技术路线。

## 2. 论文主线

这篇综述围绕 LLM 训练系统的三个核心目标展开:

- Scalability: 如何把训练扩展到成千上万张 GPU/AI 加速器，同时保持正确性和性能。
- Efficiency: 如何提高硬件利用率，减少计算、通信和内存浪费。
- Reliability: 如何在持续数周甚至数月的大规模训练中检测故障、恢复训练并减少损失。

阅读时可以一直带着这三个问题:

- 这个技术主要解决的是可扩展性、效率还是可靠性?
- 它优化的是基础设施、并行方式、计算、内存、通信还是容错?
- 它的代价是什么，例如实现复杂度、额外内存、通信开销、精度风险或恢复延迟?

## 3. 摘要与引言笔记

### 3.1 摘要要点

- LLM 训练需要大规模 GPU 集群和长时间训练，因此面临 scalability, efficiency, reliability 三类挑战。
- 论文覆盖训练基础设施、并行策略、计算优化、通信优化、内存优化和系统可靠性。
- 论文也提到传统数字电路计算系统可能难以继续满足 LLM 计算需求，因此未来可能需要光计算、光网络等新型方案。

### 3.2 Introduction

#### 背景

- LLM 的能力提升与模型规模、数据规模、计算规模密切相关。
- 大模型训练往往会长期独占大规模 GPU 集群，例如论文中提到 LLaMA-3 预训练需要约 54 天、16K H100-80GB GPU。

#### 核心挑战

- Scalability:
- Efficiency:
- Reliability:

#### 我的理解

-

#### 疑问

-

## 4. 章节结构总览

| 章节 | 主题 | 我需要重点理解的问题 |
|---|---|---|
| Section 2 | Background | LLM 训练 workload 和传统 DL 训练有什么不同? |
| Section 3 | Infrastructure | 大规模训练需要什么样的加速器、网络、存储和调度系统? |
| Section 4 | Parallelism Schemes | 数据并行、张量并行、流水线并行、序列并行、专家并行分别解决什么问题? |
| Section 5 | Computation Optimizations | 如何提升算子执行效率和数值精度效率? |
| Section 6 | Memory Optimizations | 如何降低激活、参数、优化器状态等带来的显存压力? |
| Section 7 | Communication Optimizations | 如何减少分布式训练中的通信开销? |
| Section 8 | Fault Tolerance | 长时间大规模训练如何应对故障、异常和恢复? |
| Section 9 | Conclusion and Outlooks | 未来方向是什么? |

## 5. Background

### 5.1 Transformer-based LLMs

#### 本节解决的问题

- Transformer-based LLM 的基本结构是什么?
- Attention 和 FFN 在训练系统中分别带来哪些计算/内存瓶颈?
- MHA, MQA, GQA, MLA, MoE 等结构变化对训练效率有什么影响?

#### 关键概念

- Transformer layer:
- Attention:
- FFN:
- MHA:
- MQA:
- GQA:
- MLA:
- MoE:

#### 我的理解

-

#### 疑问

-

### 5.2 LLM Training Workloads Characteristics

#### 论文列出的特点

- Homogeneous Model Architecture:
- Unprecedented Scale and Training Duration:
- Specialized Software Optimization:
- Shift in Training Paradigm:

#### 我的理解

-

#### 疑问

-

### 5.3 LLM Training Challenges

#### Scalability

-

#### Efficiency

-

#### Reliability

-

#### 可记忆总结

- Scalability 关注“能不能扩到足够大”。
- Efficiency 关注“扩上去后资源有没有被充分利用”。
- Reliability 关注“长时间跑训练时能不能稳定、不崩、可恢复”。

### 5.4 Related Survey

#### 本文和已有综述的区别

-

## 6. Infrastructure for LLM Training

### 6.1 AI Accelerators

#### NVIDIA GPU

- Ampere:
- Hopper:
- Blackwell:
- CUDA:
- HBM:

#### Other AI Accelerators

- AMD GPU / ROCm:
- Intel Gaudi:
- Google TPU:
- Graphcore IPU:
- Cerebras CS-2:

#### 我的理解

-

#### 疑问

-

### 6.2 Network Infrastructure

#### Chip-to-Chip Communications

- PCIe:
- NVLink:
- NVSwitch:
- Infinity Fabric:
- TPU ICI:

#### Node-to-Node Communications

- RDMA:
- GPUDirect-RDMA:
- InfiniBand:
- RoCE:
- iWARP:

#### Network Topology

- Clos:
- Torus:
- Dragonfly / Dragonfly+:
- Rail-Optimized:
- HPN:
- Rail-Only:
- BiGraph:
- HammingMesh:
- Reconfigurable topology:

#### Load Balancing & Congestion Control

- ECMP:
- Packet spraying:
- PFC:
- TIMELY:
- DCQCN:
- HPCC:

#### 我的理解

-

#### 疑问

-

### 6.3 Storage

#### Checkpoint Storage

- 为什么 checkpoint 对 LLM 训练很关键:
- 存储系统的瓶颈:
- 代表系统:

#### Training Data Storage

- 训练数据读取的瓶颈:
- 分布式文件系统/对象存储:
- 缓存与数据管线:

#### 我的理解

-

#### 疑问

-

### 6.4 Scheduling

#### Workload Scheduling

- 调度对象:
- 目标:
- 代表系统:

#### Resource Scheduling

- GPU/网络/存储资源如何分配:
- 资源碎片与资源利用率:

#### 我的理解

-

## 7. Parallelism Schemes for LLM Training

### 7.1 Hybrid Parallelism

#### Data Parallelism

- 基本思想:
- 优点:
- 瓶颈:
- 适用场景:

#### Tensor Parallelism

- 基本思想:
- 优点:
- 瓶颈:
- 适用场景:

#### Pipeline Parallelism

- 基本思想:
- Bubble 问题:
- Micro-batch:
- 适用场景:

#### Sequence Parallelism

- 基本思想:
- 解决的问题:
- 适用场景:

#### Expert Parallelism

- 基本思想:
- 和 MoE 的关系:
- 通信/负载均衡问题:

#### 小结表

| 并行方式 | 切分对象 | 主要收益 | 主要代价/瓶颈 |
|---|---|---|---|
| Data Parallelism | batch | 简单、易扩展 | 梯度同步通信 |
| Tensor Parallelism | 张量/矩阵计算 | 单层模型可横向扩展 | 层内通信频繁 |
| Pipeline Parallelism | 模型层 | 支持超大模型 | pipeline bubble |
| Sequence Parallelism | sequence/token | 降低激活内存 | 需要额外通信 |
| Expert Parallelism | experts | 支持 MoE 稀疏扩展 | 路由和负载均衡 |

### 7.2 Auto Parallelism

#### General Framework

-

#### Transformer-Specific Framework

-

#### 我的理解

-

### 7.3 Heterogeneous Parallelism

#### Heterogeneous Hardware

-

#### Heterogeneous Model

-

## 8. Computation Optimizations

### 8.1 Operator Optimizations

#### Manually Optimized Attention Operator

- FlashAttention:
- FlashAttention-2:
- 核心思想:
- 节省了什么:

#### Automatic Optimizations via Compilers

- 编译器优化目标:
- 算子融合:
- 自动调度:
- 代表系统:

#### 我的理解

-

### 8.2 Mixed-Precision Training

#### 16-Bit Floating Point

- FP16:
- BF16:
- Loss scaling:

#### Sub-8-Bit Floating Point

- FP8:
- 优点:
- 风险:

#### Low-Bit Fixed Point

- INT8 / lower-bit:
- 训练中使用低精度的难点:

#### 我的理解

-

## 9. Memory Optimizations

### 9.1 Activation Recomputation

#### Static Evicting

-

#### Dynamic Evicting

-

#### 我的理解

-

### 9.2 Redundancy Reduction

#### Fully Sharding

- ZeRO:
- FSDP:
- 参数/梯度/优化器状态切分:

#### Partially Sharding

-

#### 我的理解

-

### 9.3 Defragmentation

#### Tensor-based Defragmentation

-

#### VMM-based Defragmentation

-

### 9.4 Offloading

#### CPU Offloading

-

#### Dynamic Offloading

-

#### SSD Offloading

-

#### 小结表

| 方法 | 节省什么内存 | 代价 | 适用场景 |
|---|---|---|---|
| Activation Recomputation | 激活 | 额外计算 | 显存不足但算力相对充足 |
| Sharding | 参数/梯度/优化器状态 | 通信复杂度 | 大规模数据并行 |
| Defragmentation | 显存碎片 | 管理复杂度 | 长时间训练/动态分配 |
| Offloading | GPU 显存 | PCIe/CPU/SSD 传输开销 | 超大模型或显存紧张 |

## 10. Communication Optimizations

### 10.1 Collective Communication

#### Pre-Defined Collective Communication Algorithm

- AllReduce:
- AllGather:
- ReduceScatter:
- Broadcast:

#### Synthesized Collective Communication Algorithm

-

### 10.2 Communication Scheduling

#### FIFO-based Scheduling

-

#### Priority-based Scheduling

-

#### Decomposition-based Scheduling

-

### 10.3 In-Network Aggregation

#### Ethernet-based Aggregation

-

#### Infiniband-based Aggregation

-

#### 我的理解

-

## 11. Fault Tolerance

### 11.1 LLM Failure Analysis

#### 常见故障类型

- Hardware failures:
- Network issues:
- Software errors:
- Stragglers:
- Silent errors:
- Training hang:

#### 我的理解

-

### 11.2 Anomaly Detection

#### Statistical Monitoring

-

#### Proactive Validation

-

### 11.3 Checkpoint-Based Recovery

#### Persistent Checkpointing

- 优点:
- 代价:
- 关键指标:

#### In-Memory Checkpointing

- 优点:
- 代价:
- 适用场景:

### 11.4 Checkpoint-Free Recovery

#### Live Migration

-

#### Module Redundancy

-

#### 小结

- Checkpoint-based recovery 依赖保存状态，恢复更稳但可能有 I/O 开销。
- Checkpoint-free recovery 尝试减少 checkpoint 依赖，但通常需要更复杂的系统支持。

## 12. Conclusion and Outlooks

### 论文总结

-

### 未来方向

- 更高效的 AI 加速器:
- 更高性能/更低延迟的网络:
- 光计算与光网络:
- 更自动化的并行策略:
- 更强的容错与异常检测:

### 我的评价

-

## 13. 术语表

| 术语 | 中文理解 | 备注 |
|---|---|---|
| MFU | Model FLOPs Utilization，模型计算利用率 | 衡量训练系统效率 |
| HBM | High Bandwidth Memory，高带宽显存 | GPU/加速器关键资源 |
| RDMA | Remote Direct Memory Access | 绕过 CPU/OS 的高速节点间通信 |
| GPUDirect-RDMA | GPU 直接跨节点通信 | 减少 CPU 参与 |
| RoCE | RDMA over Converged Ethernet | 基于以太网的 RDMA |
| InfiniBand | 高性能网络技术 | HPC/大规模训练常用 |
| ZeRO | Zero Redundancy Optimizer | 减少数据并行中的状态冗余 |
| FSDP | Fully Sharded Data Parallel | 全切分数据并行 |
| MoE | Mixture-of-Experts | 稀疏激活专家模型 |

## 14. 代表系统/方法索引

| 系统/方法 | 所属主题 | 解决的问题 | 我是否理解 |
|---|---|---|---|
| Megatron-LM | Parallelism | 混合并行训练大模型 |  |
| DeepSpeed | Memory/Parallelism | ZeRO、offloading 等 |  |
| Alpa | Auto Parallelism | 自动并行策略搜索 |  |
| FlashAttention | Computation/Memory | IO-aware attention 优化 |  |
| ZeRO | Memory | 切分优化器状态/梯度/参数 |  |
| FSDP | Memory/Parallelism | 参数全切分 |  |

## 15. 读论文时的固定记录模板

### 今日阅读范围

- 日期:
- 页码/章节:
- 阅读耗时:

### 本次读到的核心问题

-

### 本次读到的关键方法

-

### 我觉得重要的句子

> 

### 我自己的解释

-

### 仍然不懂的问题

-

### 需要回查的概念/论文/系统

-

## 16. 待读问题清单

- [ ] Scalability, Efficiency, Reliability 三者之间有什么 trade-off?
- [ ] 为什么 LLM 训练比传统 DNN 训练更强调系统可靠性?
- [ ] 各类并行方式之间如何组合?
- [ ] ZeRO/FSDP 与 tensor parallelism、pipeline parallelism 的关系是什么?
- [ ] FlashAttention 为什么既是计算优化也是内存优化?
- [ ] checkpoint 的 I/O 开销在大规模训练中为什么会成为瓶颈?
- [ ] fault tolerance 和 anomaly detection 在 LLM 训练系统中分别承担什么角色?
