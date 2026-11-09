---
req: REQ-KVP-04
module: kv-cache-pooling
title: PE 前缀命中块异步 KV Get 与调度交接
status: 草稿
updated: "2026-11-09"
---
# REQ-KVP-04 PE 前缀命中块异步 KV Get 与调度交接

## 1. 引言

### 1.1 背景

当前 DeepInfer 的 `Scheduler` 以 `waiting`、`prompted` 和 `running` 管理请求。`DeepInfer/csrc/engine/scheduler.hpp` 中 `_schedule_prefill()` 遇到队首不可调度请求会停止扫描，`_has_prefill_waiting()` 也只检查队首。直接把 Store `get` 未完成的请求留在队首，会阻塞后续请求。

vLLM 的远端 KV 流程将“外部 block 分配、异步加载、完成确认、恢复 Prefill”分成不同阶段；vllm-ascend 的 worker 在 Python 侧执行异步 Store 操作。本需求沿用该生命周期，并适配 DeepInfer 的 C++ Scheduler 和 Python `LLMEngine.step()` 边界。

### 1.2 目标与范围

- 支持根据上游提供的命中 token 数异步拉取 prefix block。
- 拉取期间继续调度其他可执行请求。
- Store 完成后安全地恢复请求，并只计算未命中的 suffix。
- 在 Store miss、短读、取消、超时和资源不足时释放或保留正确的 HBM block。

## 2. 设计说明

### 2.1 总体流程

![异步 KV Get 流程](img/async-kv-get-flow.svg)

黄色虚线表示后台 Store 路径。Scheduler 提交 `KVLoadTask` 后继续调度其他请求；下一轮 `LLMEngine.step()` 先调用 worker 的 completion poll，再将完成事件交给 Scheduler 应用。

流程顺序如下：

1. 上游在 `add_sequence()` 前完成前缀匹配，向 DeepInfer 传入命中 token 数。
2. `Scheduler` 将请求放入 `waiting`，只保存命中结果，不提前占用 HBM。
3. 调度接纳时校验命中结果并分配 prefix block，生成 `KVLoadTask`，请求转入 `kv_loading`。
4. Python worker 依据请求 token、实际 block size、布局版本和 rank 拓扑生成 key，将每个 block 作为一个 Store item 提交到后台线程。
5. Store 直接向已注册的 `gpu_blocks` HBM segment 写入数据。
6. 下一轮 `LLMEngine.step()` 发起 completion poll，Scheduler 主线程校验 `load_id`、请求状态和各 rank 的结果。
7. 成功请求进入优先处理的 `kv_ready` 队列；suffix block 分配成功后直接进入当前或下一轮 Prefill，不追加到普通 `waiting` 队尾。
8. 失败请求按连续有效前缀截短，或整体按 miss 重新 Prefill；后台 Store 未完成前不得释放目标 block。

### 2.2 队列与状态

新增两个 Scheduler 专用队列：

```text
kv_loading    prefix block 已分配，Store get 进行中
kv_ready      Store get 已完成，等待 suffix block 分配
```
vLLM 使用 `WAITING_FOR_REMOTE_KVS` 表示异步远端 KV 尚未完成，并区分“等待启动 load”和“已经持有 KV block”两个队列。DeepInfer 复用这一语义：

### 2.3 失败、取消和资源不足

- **逐 block 失败**：按 block 顺序保留从第一个 block 开始的连续成功前缀；失败 block 及后续 block 释放并按 miss 计算。
- **只有请求级结果**：无法判断连续前缀时，整次按 miss 处理。
- **Store 连接异常**：由 Mooncake backend 的重连、重注册和熔断逻辑处理；Scheduler 只接收失败结果，不等待重连。

### 2.4 与现有 DeepInfer 入口的接入

当前入口为：

```text
LLMEngine.step()
    → Scheduler.schedule(sequence_metadata_pool)
    → post_schedule(scheduler_outputs)
    → execute_model_step()
    → process_outputs()
```

接入顺序为：

```text
LLMEngine.step()
    → KVPoolWorker.poll_completions()
    → Scheduler.schedule(sequence_metadata_pool)
    → KVPoolWorker.submit_get_tasks(scheduler_outputs.kv_load_tasks)
    → post_schedule(scheduler_outputs)
    → execute_model_step()
```

`SchedulerOutputs.kv_load_tasks` 必须是本轮输出拥有的任务数据。Python 在当前 `step()` 内立即转换为 worker 自有对象；不得借用会在下一轮 `schedule()` 中复用的 clone vector 生命周期。

## 3. 实现边界

### 3.1 C++ Scheduler 与 Python worker

- C++ Scheduler 负责命中 block 的资源接纳、队列状态、`load_id`、完成事件和 block 生命周期。
- Python worker 负责生成 Store key、构造已注册 HBM segment、后台调用 Mooncake `get` 和投递完成事件。
- worker 不直接修改 `Sequence`、`block_table` 或 Scheduler 队列；所有状态变化在主调度线程应用。
- `SchedulerOutputs.kv_load_tasks` 必须由本轮输出拥有，worker 在当前 `step()` 内转换为自有任务对象。

### 3.2 队列优先级

调度优先级为 `kv_ready`、普通 `waiting`；`kv_loading` 不参与模型调度。`kv_ready` 等价于 vLLM 的 KV holding waiting，资源不足时保持队首，不追加到普通 waiting 队尾。

## 4. 关键流程伪代码

### 4.1 Scheduler

```text
class Scheduler:
    def apply_kv_completions(self, completions):
        # 由 LLMEngine.step() 调用；Scheduler 不主动调用 Python worker。
        for completion in completions:
            seq = self.find_sequence(completion.request_id)
            if seq is None or seq.kv_load_id != completion.kv_load_id:
                continue  # 旧事件或重复事件
            if seq.schedule.status == FINISHED_ABORTED:
                self.release_loaded_blocks(seq)
                self.kv_loading.remove(seq)
                continue

            valid_blocks = continuous_success_prefix(completion.block_results)
            if valid_blocks == 0:
                self.release_loaded_blocks(seq)
                seq.cache.num_tokens = 0
                seq.prefix_hit_tokens = 0
                seq.schedule.status = WAITING
                self.kv_loading.remove(seq)
                self.waiting.push_front(seq)  # 完整 miss 优先重新 Prefill
                continue

            self.truncate_loaded_blocks(seq, valid_blocks)
            seq.prefix_hit_tokens = valid_blocks * self.block_size
            seq.schedule.status = WAITING
            self.kv_loading.remove(seq)
            self.kv_ready.push_back(seq)  # 对应 vLLM kv_holding_waiting

    def admit_kv_loads(self, outputs):
        for seq in self.waiting:
            hit_tokens = seq.prefix_hit_tokens_from_upstream
            if hit_tokens == 0:
                continue
            if hit_tokens % self.block_size != 0:
                self.mark_prefix_miss(seq)
                continue

            block_count = hit_tokens // self.block_size
            if not self.block_manager.can_allocate_nolock(block_count):
                break  # 保持 waiting 队首语义

            local_blocks = self.block_manager.allocate_nolock(block_count)
            self.pin(local_blocks)
            seq.cache.block_table.assign(local_blocks)
            seq.cache.num_tokens = hit_tokens
            seq.prefix_hit_tokens = hit_tokens
            seq.kv_load_id = self.next_load_id()
            seq.schedule.status = WAITING_FOR_REMOTE_KVS
            self.kv_loading.push_back(self.waiting.pop_front())
            outputs.kv_load_tasks.push_back(self.make_task(seq, local_blocks))

    def schedule(self, sequence_metadata_pool):
        # vLLM 的 kv_holding_waiting 语义：ready 请求优先，资源不足时保持队首。
        while self.kv_ready:
            seq = self.kv_ready.front()
            suffix_blocks = self.required_suffix_blocks(seq)
            if (not self.block_manager.can_allocate_nolock(suffix_blocks)
                    or not self.prefill_budget_allows(seq)):
                break
            self.allocate_suffix_blocks(seq, suffix_blocks)
            self.kv_ready.pop_front()
            self.schedule_prefill(seq)

        outputs = SchedulerOutputsBuilder()
        self.admit_kv_loads(outputs)
        outputs.extend(self.schedule_normal_prefill_or_decode(sequence_metadata_pool))
        return outputs

    # `schedule_prefill` 继续复用现有 chunked-prefill 路径；若 suffix 未在一轮完成，
    # 请求按原有 chunk 状态在后续轮次继续执行，不重新提交 Store get。
```

### 4.2 KVPoolWorker

```text
class KVPoolWorker:
    def submit_get_tasks(self, tasks):
        for task in tasks:
            self.executor.submit(self._get_one_task, task)

    def _get_one_task(self, task):
        items = []
        for index, local_block_id in enumerate(task.local_block_ids):
            # key 在 PE 侧生成；C++ Scheduler 不保存完整 Store key。
            key = build_store_key(task.request_tokens, task.block_size,
                                  task.layout_version, self.rank_info, index)
            address = self.gpu_blocks.data_ptr() + local_block_id * task.block_bytes
            items.append(GetItem(key, [(address, task.block_bytes)]))

        # backend 负责短读、对象不存在、重连和熔断，并返回逐 item 结果。
        result = self.mooncake_store.get(items)
        self.completion_queue.put(
            KVLoadCompletion(task.request_id, task.kv_load_id, result)
        )

    def poll_completions(self):
        # 由 LLMEngine.step() 发起；这里只读取完成队列，不修改 Scheduler 状态。
        return self.completion_queue.drain()
```

### 4.3 LLMEngine 调用顺序

```text
LLMEngine.step():
    # Engine 发起 poll，并把结果交给 C++ Scheduler 应用。
    completions = kv_worker.poll_completions()
    scheduler.apply_kv_completions(completions)

    seq_metadata, scheduler_outputs = scheduler.schedule(sequence_metadata_pool)

    # SchedulerOutputs 在 Python 当前 step 内转换为 worker 自有 task。
    kv_worker.submit_get_tasks(scheduler_outputs.kv_load_tasks)
    post_schedule(scheduler_outputs)
    execute_model_step(seq_metadata, scheduler_outputs)
    process_model_outputs(scheduler_outputs)
```

## 5. ST 测试用例

测试用例只描述验证计划，不代表已经执行或通过。

| 用例编号     | 类型 | 测试目的                     | 测试步骤                                                                          | 测试点 / 预期结果                                                                          |
| ------------ | ---- | ---------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| TC-KVP-04-01 | 正常 | 验证完整命中异步恢复         | 提交对齐的命中数；保持 Store 未完成；再返回全部 block 成功并执行下一轮 step       | get 期间其他请求继续调度；完成请求进入 `kv_ready` 优先队列并执行 suffix Prefill。        |
| TC-KVP-04-02 | 正常 | 验证无命中复用原有 Prefill   | 提交 `hit_tokens == 0` 的请求并执行调度                                         | 不分配 prefix block、不生成异步 task，直接走现有 Prefill。                                 |
| TC-KVP-04-03 | 并发 | 验证优先队列和投票边界       | 让一个请求完成 get、另一个请求仍在 `kv_loading`，分别设置 suffix 资源足够和不足 | `kv_ready` 优先处理；资源不足时保持队首；`kv_loading` 不参与 ALL/ANY，ready 后才参与。 |
| TC-KVP-04-04 | 异常 | 验证部分失败和短读回退       | 多 block 请求中间 block 返回短读或失败                                            | 只保留连续有效前缀，后续 token 走普通 Prefill；不使用失败 block 后的数据。                 |
| TC-KVP-04-05 | 异常 | 验证整体 miss 与资源清理     | 让 Store 返回请求级失败或首 block 失败                                            | prefix block 释放且不重复释放，命中数清零，请求优先回到 waiting 重新计算。                 |
| TC-KVP-04-06 | 异常 | 验证取消、重复和过期完成事件 | get 在途时取消请求，再注入完成、重复完成和旧 `load_id` 事件                     | 事件被丢弃，block 只释放一次，不修改已取消或已完成请求。                                   |
