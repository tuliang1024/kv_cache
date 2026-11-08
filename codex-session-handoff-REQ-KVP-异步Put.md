# Codex 会话交接记录：REQ-KVP-06 异步 Block Put

> 用途：将当前会话的背景、结论和文件状态迁移到另一台电脑的 Codex。
> 生成日期：2026-11-09

## 1. 任务背景

为 DeepInfer 设计 Prefill/Decode KV cache 完整 block 的异步 Put 方案，参考：

- vLLM：`/Users/l/L/Code/TuringProject/D/vllm`
- vLLM Ascend：`/Users/l/L/Code/TuringProject/D/vllm-ascend`
- DeepInfer：`/Users/l/L/Code/TuringProject/D/DeepInfer`
- 上游 Mooncake Store 设计：`/Users/l/L/Code/TuringProject/D/设计文档/mooncake_store.md`
- 方案草稿：`/Users/l/L/Code/TuringProject/D/设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`

当前工作重点是设计文档草稿，不是直接实现代码。

## 2. 已确认的方案结论

### 2.1 Prefill

- 普通 Prefill 可视为一次完成的 prompt chunk。
- Chunk Prefill 只有最后一个 prompt chunk 才进入最终 Put 流程。
- 模型执行前确定需要保存的完整 block，并执行 `pin_blocks`。
- 模型执行完成后再将 PutTask 提交给 KV worker。
- Prefill 尾部不足一个 block 的 KV 不单独写入 Store。

### 2.2 Decode

- Decode 根据已提交 token 数和 `block_size` 计算完整 block。
- 通过 `next_unsaved_block` 避免同一 block 重复提交。
- Prefill 尾部未满 block 后续由 Decode 填满后再提交。
- 请求结束时只补交已经完整但尚未提交的 block。

### 2.3 生命周期

- PutTask 必须保存独立的 `block_ids` 快照，不能异步读取会被清空的 `seq.cache.block_table`。
- 提交前调用 `BlockManager.pin_blocks()`，Store worker 完成后由 Engine 轮询 completion 并调用 `unpin_blocks()`。
- Put 期间 Scheduler 可以继续释放请求自身引用，但 Put 引用必须保持到 completion 被消费。
- Store Put 失败不阻塞推理，失败结果记录并释放 pin。

### 2.4 DeepInfer 主流程接入

已确认的调用关系：

```text
LLMEngine.step()
  -> Scheduler.schedule()
  -> Worker.execute_model_step()
  -> Worker.pre_sampler_sync()
  -> Worker._post_process_step()
  -> LLMEngine.post_process()
```

已确认代码位置：

- Engine callback 注册：`DeepInfer/deepinfer/engine/llm_engine.py:399`
- Decode 保存入口：`DeepInfer/deepinfer/engine/llm_engine.py:830-846`
- Prefill `post_process`：`DeepInfer/deepinfer/engine/llm_engine.py:984-1015`
- Worker post process：`DeepInfer/deepinfer/worker/worker.py:248-271`
- Sequence block table 释放：`DeepInfer/csrc/engine/scheduler.hpp:738-745`

设计中的新增钩子：

- `pe_prepare_save_kvcache()`：准备 Prefill 完整 block、复制 block ID、pin，不调用 Store。
- `pe_do_save_kvcache()`：在模型执行完成后的 `post_process()` 中提交 Prefill PutTask。
- `do_save_kvcache()`：Decode 计算新增完整 block、pin 并提交 PutTask。
- `poll_put_completions()`：Engine 轮询 worker completion，执行 unpin 和结果记录。

### 2.5 Mooncake Store 接口

上游设计中对象类型为 `PutItem`，调用形式为：

```python
items = [
    PutItem(key=key, segments=[(addr, size)])
    for key, addr, size in zip(keys, addrs, sizes, strict=True)
]
result = mooncake_store.put(items)
```

`MooncakeStoreClient.put(items)` 对外返回按 item 对齐的成功结果；底层 `batch_put_from_multi_buffers` 的 `0/-705` 归一为成功。当前 Put 设计不在 worker 中调用 Store `get`。

## 3. 已生成文件

主设计文档：

- `设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`

串讲备忘：

- `设计文档/REQ-KVP-06-block-put/设计串讲备忘.md`

异步 Put 时序图：

- `设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`

当前时序图表达：

1. Engine 确定完整 block。
2. 复制 `block_ids` 并构造 PutTask。
3. 调用 `pin_blocks()`。
4. pin 完成后调用 `submit_put(task)`。
5. KVPoolWorker 等待 `ready_event` 并调用 `Store.put(items)`。
6. Engine 继续调度其他请求。
7. 下一次 step 轮询 completion，完成 `unpin_blocks()`。

SVG 已通过：

```bash
xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg
rsvg-convert -o /tmp/async-block-put-sequence-final.png \
  设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg
```

## 4. 文档当前结构

主文档保留 1 至 5 章：

1. 引言
2. 设计说明
3. DeepInfer 主流程接入
4. 关键流程伪代码
5. 测试用例

已删除摘要、非目标、约定用语、独立接口章节、依赖约束和参考文献。遗留讨论放在 `设计串讲备忘.md`。

## 5. 继续工作时的约束

- 中文输出。
- 设计逻辑优先参考 vLLM/vLLM Ascend 和 DeepInfer 真实代码，原方案草稿只作辅助。
- 先确认代码事实，再修改文档或 SVG。
- 不要把普通 Prefill 误写成只有“最后一个 prompt chunk”。
- 不要把 Engine 的保存回调、KV worker 的 Store 操作和 Scheduler 主流程混写在同一个职责边界内。
- 不要将异步 Put 描述成阻塞 Scheduler；Engine 应继续执行后续 step，并在后续轮询 completion。

## 6. 在另一台电脑上的使用方法

### 方法 A：直接把交接文件作为首条消息提供给 Codex

将本文件和相关仓库同步到另一台电脑后，在新 Codex 会话中发送：

```text
请先读取 /path/to/设计文档/codex-session-handoff-REQ-KVP-异步Put.md，
按其中的背景、结论和文件状态继续工作。先检查文件是否存在，再执行我后续的修改要求。
```

随后把 `/path/to` 替换为新电脑上的实际仓库路径。

### 方法 B：让 Codex 从交接文件恢复上下文

```text
请读取以下会话交接文件并恢复工作上下文：
/Users/l/L/Code/TuringProject/D/设计文档/codex-session-handoff-REQ-KVP-异步Put.md

恢复后先报告：
1. 当前已完成内容；
2. 相关文件路径；
3. 尚未完成的事项；
然后等待我的下一条修改指令。
```

### 重要说明

Codex 会话本身通常不能通过普通文件完整迁移，包括平台内部消息 ID、工具状态和打开的 IDE 标签页。该文件保存的是可复现的工作上下文；迁移后仍需确保 vLLM、vLLM Ascend、DeepInfer 和设计文档目录位于可访问路径。

