# SGLang 遗留问题

## RadixAttention 的 KV Cache 淘汰机制

**问题：**SGLang 的 `evict blocks` 是否使用 LRU？LRU 如何与 ref count 配合，是否只有叶节点才能成为淘汰候选？`ref_count == 0` 是否就代表叶节点？如果一个公共前缀的 `ref_count == 0`，是否说明整棵子树都没有被使用，可以直接删除？如果所有 block 的 ref count 都不为 0，又会如何处理？

**答：**SGLang 采用引用计数保护下的叶节点 LRU，**淘汰条件是“叶节点且 `ref_count == 0`”**：ref count 表示是否有运行请求使用，叶节点表示是否没有子节点，两者并不等价。由于子节点的引用会传播到祖先，公共前缀的 `ref_count == 0` 确实说明整棵子树没有运行请求使用；整棵删除虽然安全，但可能误删近期使用的分支并释放过多空间，因此系统从最旧叶子开始逐步淘汰，空间仍不足时才可能最终删完整棵子树。如果所有 block 的 ref count 都不为 0，LRU 就没有候选节点：新请求需要等待；若当前 decode 也没有足够空间，当前实现会撤回部分运行请求并释放其 KV，单个请求仍无法容纳时则返回 OOM。

## FSM 与压缩 FSM 总结

### 1. FSM 解决什么问题？

FSM 用于**约束模型输出格式**。系统将正则表达式或 JSON Schema 转换成状态机：节点代表状态，边代表允许出现的字符或字符串；生成内容只有沿合法边前进并到达终止状态，才是合法输出。

提示词中的格式要求只是软约束，模型可以不遵守；FSM 在解码阶段直接限制候选 token，属于硬约束。

### 2. FSM mask 如何工作？

一次 decode forward 会为词表中的每个 token 输出一个 logit。每个 logit 只是一个实数分数，表示当前上下文对该 token 的相对倾向；经过 softmax 后才成为概率。

运行时根据当前 FSM 状态构造覆盖整个词表的 mask：

```text
mask[i] = 0       token i 合法
mask[i] = -∞      token i 非法
概率 = softmax(logits + mask)
```

随后只从合法 token 中采样，用生成 token 的字符串更新 FSM 状态，再为下一步生成新的 mask。FSM 不修改模型参数，只在模型计算 logits 与采样之间动态限制候选集合。

### 3. 一个完整例子

要求输出 `{"grade":"A"}`，其中评分只能为 `A/B/C/D`：

```text
q0 --固定文本 {"grade":"--> q1
q1 --A/B/C/D-----------> q2
q2 --固定文本 "}--------> q3（终止状态）
```

在 `q1`，模型可能给出：

| token | 原 logit | mask 后 |
|---|---:|---:|
| `A` | 1.0 | 1.0 |
| `B` | 2.0 | 2.0 |
| `X` | 4.0 | `-∞` |

虽然模型最想生成 `X`，但其概率被置为 0。假设最终选择 `B`，FSM 转到 `q2`，补上固定结尾并到达 `q3`，最终输出 `{"grade":"B"}`。

### 4. 普通 FSM 与压缩 FSM

- **普通 FSM**：即使一段文本只有唯一合法写法，也逐 token 调用模型和应用 mask。
- **压缩 FSM**：把连续的唯一分支合并成一条边，通过 jump-forward 一次处理多个确定 token；只有遇到分支时才让模型选择。

压缩 FSM 不改变格式和长度约束，只减少不必要的逐 token forward。由于 FSM 通常建立在字符或字符串上，而模型使用 token，SGLang 还需要重新分词，处理字符边界与 token 边界不一致的问题。

可以把 jump-forward 近似理解为一次**增量 prefill/extend**：运行时把格式已经确定的字符串重新分词，作为已知的续写 token 追加到当前序列，并复用已有 KV Cache，只对新增部分执行一次多 token forward。它不是重新 prefill 整个上下文，也不是让模型重新选择这些确定 token；模型仍需计算新增 token 在各层的表示和 KV。之后遇到有多个合法选择的状态，再恢复普通 decode 和 FSM mask。

### 5. FSM 如何控制长度？

FSM 通过“何时能够到达终止状态”间接控制长度：`\d{4}` 必须生成 4 个数字，`\d+` 则允许循环，长度不固定。`max_new_tokens` 是 FSM 之外的硬上限。字符数量也不一定等于 token 数量。

### 6. 为什么 SGLang 不能在闭源 API 外部执行 FSM？

FSM 需要在每个解码步骤读取和修改 logits。自托管模型的运行时可以做到；黑盒 API 客户端通常只能发送提示词并接收结果，无法介入内部解码。即使把固定 JSON 前缀放进提示词，也只能提前 prefill 开头部分，不能动态屏蔽后续非法 token。

闭源模型本身仍然可以使用 FSM：API 提供商可以接收 JSON Schema、正则或 grammar，并在服务器内部执行约束解码。此时控制权属于 API 提供商，而不是外部的 SGLang。

## SGLang 代码模板与 Agent 分工总结

论文 Figure 2 展示的是一个 **LM program**，不一定是自主 Agent。SGLang 原语代码位于 Agent 的**工作流编排/模型调用层**，用于描述模型调用之间的控制结构。

| 角色 | 主要职责 |
|---|---|
| 开发者或高层编译器 | 编写可复用的工作流模板 |
| **规则路由器或规划模型** | **根据用户问题选择模板并填写参数** |
| 任务模型 | 在 `gen/select` 位置生成判断和文本 |
| SGLang Runtime | 执行模板，并优化并行、缓存和约束解码 |

模板抽象的是“串行、并行、分支、循环、合并、约束输出”等**任务流结构**；提示词、评审维度、工具、分支数和 schema 是动态参数。因此三维评审改为四维评审只需更换 `dimensions`，不需要新代码。

```text
用户问题 → 规则或模型路由器 → 模板 + 参数 → SGLang Runtime → 模型
```

SGLang 本身不负责选择模板。固定应用可由普通路由代码选择；通用 Agent 可让规划模型通过 tool calling 返回模板名称和参数，再由宿主程序验证并调用。

不需要预写成千上万段代码，少量参数化模板可以覆盖大量请求。只有任务流拓扑真正改变时才可能需要新模板；如果没有合适模板，系统仍可退回单次模型调用或通用 Agent 循环，只是可能无法充分利用针对该结构的多调用优化。

模型可以辅助开发者生成模板，DSPy 等高层系统也可以编译到 SGLang，但论文默认方式不是任务模型在执行过程中临时生成并运行 SGLang 源码。普通用户通常只使用已经开发好的 Agent。

## SGLang 总结

**一句话：**SGLang 是一个用于编写和高效执行 **LM Program** 的系统，由 Python 内嵌的前端语言和推理运行时组成。它不是模型，也不是完整 Agent；它把“多次模型调用的程序结构”显式写出来，再让运行时跨调用优化。

```text
用户 → Agent 的业务逻辑、状态与工具
     → SGLang 前端：描述提示词、生成、选择、分支和并行
     → SGLang Runtime 或外部模型 API
     → 模型
```

### 1. 它解决什么问题？

复杂 LLM 应用通常包含多次且相互依赖的调用、共享前缀、并行分支和结构化输出。若把每次调用当成互不相关的普通请求，不仅代码难写，还会重复计算前缀 KV Cache，并产生额外的逐 token 解码和 API 等待开销。

### 2. 前端语言做什么？

- 用 `gen`、`select`、`fork`、`join` 等原语描述模型调用和控制结构。
- 与 Python 的条件、循环和库直接配合；提示词状态 `s` 按提交顺序形成异步流。
- 支持解释器模式，也可以追踪并编译成计算图，以便进一步优化。

前端主要解决“**工作流怎样表达**”，不负责模型本身的矩阵计算。

### 3. 运行时的三项核心优化

| 优化 | 针对的瓶颈 | 核心做法 |
|---|---|---|
| **RadixAttention** | 多次调用反复 prefill 相同前缀 | 用 radix tree 保存和匹配跨请求 KV Cache，并结合 cache-aware scheduling 提高命中率 |
| **压缩 FSM** | JSON、正则等约束输出仍逐 token forward | 在分支处用 mask 限制候选；把确定的连续路径合并，并通过 jump-forward 一次处理多个已确定 token |
| **API speculative execution** | 黑盒 API 的多调用串行等待和重复输入 | 让前一次 API 调用越过停止位置继续生成；若后续提示与猜测相符，就复用多生成的内容 |

三者分别主要减少：**重复 prefill、结构化输出的 decode 次数、API 多调用等待**。

### 4. 它在 Agent 中处于哪里？

SGLang 位于 Agent 的**工作流编排和模型调用层**。Agent 的状态管理、工具实现、权限和业务规则仍由应用负责；开发者或高层编译器编写可复用模板，路由器选择模板和参数，模型只在 `gen/select` 等位置生成内容，SGLang Runtime 负责高效执行。

因此，不需要为每一种用户措辞写一段新代码；只有任务流结构真正变化时，才可能需要增加或修改模板。

### 5. 前端、运行时和 vLLM 的关系

- SGLang 前端可以调用自己的 SGLang Runtime，也可以调用 OpenAI、Anthropic 等 API。
- SGLang Runtime 也可以作为独立推理服务使用；前端提供结构提示时，两者协同优化效果更好。
- SGLang Runtime 与 vLLM 属于同一层的推理系统，可以替代使用；论文中的 SGLang 并不是运行在用户终端、辅助 vLLM 的客户端，也不是直接复用 vLLM 实现。

### 6. 什么时候最有价值？

最适合多调用、共享前缀、分支并行以及 JSON/正则约束输出。若任务只是一次普通生成，没有可复用前缀和输出约束，收益通常较小。对于黑盒 API，外部 SGLang 无法直接操作其 KV Cache 和 logits，只能使用 API speculative execution 等客户端优化。

论文实验覆盖开源模型、多模态模型和 GPT-3.5 API 等负载，报告最高 **6.4× 吞吐提升**和最高 **3.7× 延迟降低**。核心结论是：**前端暴露工作流结构，运行时利用这些结构做跨调用优化，二者协同才能获得最大收益。**

### 7. SGLang 是否只是在 vLLM 上加入 Radix Tree？

不是。论文中的 SGLang Runtime 是独立实现，vLLM 0.2.5 是对照系统，并非其代码底座。若只看开源模型运行时，RadixAttention 确实是最突出的贡献之一；但论文的整体核心是**前端语言与运行时协同设计**，运行时还包括压缩 FSM 和 API speculative execution。RadixAttention 也不只是加入一棵 radix tree，而是由 KV Cache 持久保留、前缀树匹配、LRU 淘汰和 cache-aware scheduling 共同组成。

不过，SGLang 的确采用了 **PagedAttention 所代表的分页式 KV Cache 存储思想**，这项工作由 vLLM 提出。两者职责不同：PagedAttention 解决 KV Cache **怎样用非连续页高效存放和分配**；RadixAttention 在此基础上解决这些 KV 页**属于哪个 token 前缀、能否跨调用复用、保留或淘汰哪些页**。因此可以说 SGLang 借用了 vLLM 的底层思想，但不能说 SGLang 是直接在 vLLM 系统上加了一棵树。

## 资料来源

- [SGLang 论文](../2026-07-20-SGLang-Efficient-Execution-of-Structured-Language-Model-Programs/SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.pdf)
- [RadixCache 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [请求撤回实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L2524-L2574)
