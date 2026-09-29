这里的 `In the traditional programming model` 指的是：

> 在传统的、由 CPU 主导发起 RDMA 操作的编程模式中。

它不是指“传统机器学习编程”，也不是泛指传统 CUDA，而是在对比两种 RDMA 控制方式：

```
传统模式：CPU-initiated RDMA
论文采用：GPU-initiated RDMA
```

## 原文翻译

> 在传统编程模型中，CPU 负责注册内存、准备工作请求，并在每一次通信操作中敲响 NIC 的门铃。这种模型适合大块数据传输，但不适合 EP 的 token 分发与结果合并：后者的数据量较小、通信轮次频繁，任何 CPU 参与都会直接进入解码关键路径。

## 传统 CPU 主导的 RDMA 流程

假设 GPU 0 要把 token hidden states 发送到另一台服务器的 GPU 1：

```
源 GPU 0
   ↑
CPU 准备 RDMA 请求
   ↓
本机 RNIC
   ↓ 网络
远端 RNIC
   ↓
目标 GPU 1
```

CPU 主要负责三个步骤。

### 1. 注册内存

CPU 告诉 RNIC：

> 这一段 GPU 显存允许通过 RDMA 直接访问。

注册完成后，RNIC 才知道：

- GPU Buffer 地址；
- Buffer 大小；
- 本地和远程访问权限；
- `lkey/rkey` 等访问密钥。

内存注册一般主要发生在初始化阶段，不一定每次传输都重新执行。

### 2. 准备 Work Request

CPU 为 RNIC 创建一个工作请求，描述：

```
操作类型：RDMA Write
源地址：GPU 0 的发送 Buffer
目标地址：GPU 1 的接收 Buffer
数据长度：例如 64 KB
目标连接：Rank 3 对应的 Queue Pair
访问密钥：Remote Key
```

可以理解为 CPU 给网卡填写了一张传输任务单。

### 3. Ring the NIC doorbell

CPU 把 Work Request 放入 Work Queue 后，需要写入一个特殊的 Doorbell Register，通知 RNIC：

> 队列里有新任务，可以开始传输了。

`ring the doorbell` 并不是真的按门铃，而是向网卡寄存器写入一个值。

完整控制流程是：

```
CPU 创建 Work Request
   ↓
CPU 将请求写入 Work Queue
   ↓
CPU 写 Doorbell Register
   ↓
RNIC 读取 Work Request
   ↓
RNIC 执行 GPU-to-GPU RDMA 传输
```

数据不一定经过 CPU 内存，但**通信操作仍由 CPU 发起和提交**。

## GPUDirect RDMA 不一定等于 GPU-initiated RDMA

这个区别很重要。

传统 GPUDirect RDMA 可以做到：

```
数据路径：
源 GPU → RNIC → 网络 → RNIC → 目标 GPU
```

数据绕过 CPU 内存，但控制路径仍然可能是：

```
CPU → 创建 Work Request → Ring Doorbell
```

因此：

> CPU 不搬数据，不代表 CPU 不参与发起通信。

而 GPU-initiated RDMA 进一步把控制权也交给 GPU：

```
GPU Kernel
   ↓
直接写入 GPU 可访问的 Work Queue
   ↓
GPU Kernel 直接 Ring Doorbell
   ↓
RNIC 发起 RDMA
```

## 为什么传统模式适合 Bulk Transfer？

`bulk transfer` 指**少次数的大规模数据传输**，例如：

```
一次传输 1 GB 数据
```

假设 CPU 发起请求需要几微秒，而 RNIC 传输 1 GB 需要更长时间，那么 CPU 开销占比很小：

```
CPU 提交：几微秒
网络传输：数百微秒或更长
```

所以传统 CPU 主导方式没有明显问题。

## 为什么不适合 EP Dispatch/Combine？

MoE Decode 中的通信特点是：

```
数据量较小
通信频率很高
每个 MoE 层都需要 Dispatch 和 Combine
每个 Decoding Step 都会重复
```

例如：

```
MoE Layer 1：Dispatch → Combine
MoE Layer 2：Dispatch → Combine
...
MoE Layer 60：Dispatch → Combine
```

如果每轮都需要：

```
GPU 计算路由
→ 通知 CPU
→ CPU 准备 Work Requests
→ CPU Ring Doorbells
→ GPU 等待通信
```

CPU 就会进入每个 decoding step 的关键路径，增加：

- 延迟；
- CPU/GPU 同步；
- kernel 间空隙；
- 调度抖动；
- CUDA Graph 集成难度。

## 论文为什么在这里介绍它？

EEP 需要 CUDA Graph 内的 dispatch kernel 根据当前状态动态决定：

```
Rank 0：active，发送
Rank 1：inactive，跳过
Rank 2：active，发送
```

如果仍采用 CPU 主导模式，GPU 每轮都可能需要回到 CPU 查询：

```
现在有哪些 Rank 活跃？
应该向哪些 Peer 发送？
```

这会把 CPU 再次放入 decode critical path，也破坏 CUDA Graph 的稳定快速执行。

GPU-initiated RDMA 则允许：

```
CUDA Kernel
   ↓
读取 GPU 上的 Peer Table
   ↓
只为 active peers 创建 Work Queue Entries
   ↓
GPU 直接 Ring Doorbell
   ↓
RNIC 执行传输
```

Rank 故障后，只需修改 GPU Peer Table：

```
故障前：
Rank 3.active = 1

故障后：
Rank 3.active = 0
```

下一次 CUDA Graph replay 时，kernel 会直接跳过 Rank 3，不需要：

- CPU 每轮介入；
- 重新创建通信执行路径；
- 重新捕获 CUDA Graph。

所以论文提到“传统编程模型”，是为了引出这个关键结论：

> 传统 CPU-initiated RDMA 虽然能让数据绕过 CPU 内存，却仍让 CPU 负责发起每次通信；而 EEP 需要 GPU kernel 根据设备端的可变成员状态直接发起 RDMA，才能同时满足低延迟、CUDA Graph 稳定性和动态故障绕过。

- 解释 RDMA Work Queue
- 对比两种 RDMA 路径
- 继续讲解 Figure 3