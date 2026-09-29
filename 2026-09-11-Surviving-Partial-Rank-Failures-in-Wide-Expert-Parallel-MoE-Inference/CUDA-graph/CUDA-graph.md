# CUDA graph

## 1. 什么是 kernel launch overhead？

==GPU 上的**矩阵乘法、RMSNorm、Attention、激活函数**等操作，最终都会被实现为一个或多个 CUDA kernel。==

如果不使用 CUDA Graph，一次模型前向大致是：

```
CPU
├─ launch RMSNorm kernel
├─ launch QKV GEMM kernel
├─ launch Attention kernel
├─ launch Output GEMM kernel
├─ launch Router kernel
├─ launch Dispatch kernel
├─ launch Expert GEMM kernel
└─ launch Combine kernel
        ↓
GPU 依次执行
```

CPU 每启动一个 kernel，都需要完成：

- 准备 kernel 参数；
- 检查依赖关系；
- 向 CUDA Driver 提交任务；
- 将任务放入 CUDA Stream；
- 触发 GPU 执行。

单次启动可能只有数微秒，但一个 Transformer 模型每个 decoding step 可能包含数百、数千次 kernel launch。

## 2. 为什么 Decode 特别在意启动开销？

Decode 每一步通常只为每个请求生成一个 token：

```
第 t 个 token
   ↓ 完整模型前向
第 t+1 个 token
   ↓ 完整模型前向
第 t+2 个 token
```

由于每一步**计算规模相对较小**，GPU kernel 本身可能执行得很快，**CPU 的 kernel launch 时间就会占据明显比例**。

例如：

```
Kernel 实际执行：10 μs
CPU launch 开销：5 μs
```

如果模型中有大量短 kernel，GPU可能频繁等待 CPU 提交下一项任务，形成 CPU launch bottleneck。

Prefill 的矩阵计算规模更大，单个 kernel 执行时间更长，因此启动开销所占比例通常没有 Decode 那么突出。

## 3. CUDA Graph 如何解决？

==CUDA Graph 允许系统先把一次完整执行流程捕获成有向无环图：==

==为什么说CUDA Graph与CPU启动相比更高效？是因为CPU每次启动kernel都需要向GPU传输信息，但是CUDA graph方式只需要传输一次CUDA graph就可以了，之后GPU就按照graph执行，不需要CPU再次介入==

```
RMSNorm
   ↓
QKV GEMM
   ↓
Attention
   ↓
Router
   ↓
Dispatch → Expert GEMM → Combine
```

基本流程是：

```
第一次：
Capture → Instantiate CUDA Graph

后续每一步：
Replay CUDA Graph
Replay CUDA Graph
Replay CUDA Graph
```

CPU 不再逐个 launch 所有 kernel，而是**用一次图重放提交预先准备好的整套执行流程**：

```
传统执行：
CPU → kernel 1 → kernel 2 → kernel 3 → ...

CUDA Graph：
CPU → replay 整张图 → GPU 执行所有节点
```

这样可以显著降低：

- CPU 调度开销；
- CUDA API 调用次数；
- kernel 之间的空隙；
- 每个 decoding step 的延迟抖动。

## 4. freezing kernel structure 是什么？

`kernel structure` 指图中包含==**哪些 kernel，以及它们的顺序和依赖关系**==。

捕获时可能是：

```
Kernel A
   ├─→ Kernel B
   └─→ Kernel C
           ↓
        Kernel D
```

捕获完成后，CUDA Graph 默认认为后续 replay 仍采用这套结构。不能在某一次 replay 中随意变成：

```
Kernel A → 新增 Kernel X → Kernel D
```

在 EP 场景中，结构可能包含：

```
统计各 rank 的 token 数
    ↓
将 token dispatch 到对应 rank
    ↓
执行本地 Experts
    ↓
从各 rank combine 结果
```

==如果成员关系改变，需要增加、删除或替换通信目标，原来的图结构就可能不再适用。==

## 5. freezing launch parameters 是什么？

启动一个 CUDA kernel 时需要指定参数，例如：

```
kernel<<<grid_size, block_size, shared_memory, stream>>>(arguments);
```

包括：

- `grid size`：启动多少个线程块；
- `block size`：每个线程块有多少线程；
- 动态共享内存大小；
- kernel 参数；
- tensor shape、stride 等间接影响执行的信息。

捕获时如果系统按照 32 个 rank、固定 batch bucket 或固定 token 容量建立图，那么 replay 时通常也要满足这些预设条件。

这也是推理框架常为不同 batch size 分别捕获 CUDA Graph 的原因：

```
Batch 1  → Graph 1
Batch 2  → Graph 2
Batch 4  → Graph 4
Batch 8  → Graph 8
```

实际 batch 不匹配时，系统可能通过 padding 或选择最接近的 bucket 来重放。

## 6. freezing pointer identities 是什么？

CUDA kernel 的参数中经常包含 GPU 内存地址：

```
输入 activation 地址
输出 buffer 地址
KV Cache 地址
通信 buffer 地址
路由表地址
Peer table 地址
```

CUDA Graph 捕获的不是“以后随便找一块保存同类数据的内存”，而是具体的设备虚拟地址，例如：

```
input_ptr  = 0x1000
output_ptr = 0x5000
peer_table = 0x9000
```

重放时，这些地址通常必须继续有效。因此推理框架会预先分配固定缓冲区，每次把新的输入数据写到同一个地址，再 replay CUDA Graph。

## 7. 固定指针不等于固定数据

这是理解论文方案最重要的一点：

```
指针地址固定 ≠ 指针指向的内容固定
```

例如 CUDA Graph 始终读取：

```
peer_table_ptr = 0x9000
```

但 `0x9000` 指向的表内容可以修改：

```
故障前：
[Rank0=active, Rank1=active, Rank2=active, Rank3=active]

Rank3 故障后：
[Rank0=active, Rank1=active, Rank2=active, Rank3=inactive]
```

因此可以做到：

```
Graph 结构不变
指针地址不变
表中内容可变
```

这正是 EEP 的关键设计思路：**让 CUDA Graph 捕获稳定的指针，而把 membership 和 routing 信息放在这些指针指向的可变 GPU 表中。**

## 8. 为什么 rank 故障会破坏传统 CUDA Graph？

假设图捕获时有 4 个 ranks：

```
Expert 0～7   → Rank 0
Expert 8～15  → Rank 1
Expert 16～23 → Rank 2
Expert 24～31 → Rank 3
```

CUDA Graph 中可能已经固化：

- 发往 Rank 3 的通信 kernel；
- Rank 3 的通信 buffer 地址；
- Expert 24～31 的位置；
- 4 个 rank 对应的同步关系；
- 按 4-rank membership 计算出的路由元数据。

Rank 3 故障后，如果直接把它删除：

```
Rank set：4 → 3
Expert placement：改变
路由目标：改变
通信 buffer：改变
同步关系：改变
```

原 CUDA Graph 与实际运行状态不一致。传统方案通常只能：

```
销毁旧 Graph
重新建立通信和内存状态
重新执行 warmup
重新 capture
重新 instantiate
```

而这个过程可能持续数秒甚至更长。

## 9. EEP 如何化解这个冲突？

传统方式把 membership 写死在图结构中：

```
CUDA Graph
└─ 直接包含 Rank 0、1、2、3 的具体状态
```

EEP 改为间接引用：

```
CUDA Graph
└─ 固定地址的 Peer/Route Table
      ├─ 当前 active ranks
      ├─ Expert → Rank 映射
      └─ 通信地址与状态
```

故障后只修改表内容：

```
旧表：Expert 26 → Rank 3
新表：Expert 26 → Rank 1
```

CUDA Graph 本身不变：

```
同一个 kernel
同一个执行结构
同一个 table 指针
不同的 table 内容
```

所以健康 ranks 不需要重新捕获 CUDA Graph。