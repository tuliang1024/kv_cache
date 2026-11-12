
有。当前最稳妥的方案是：**不要依赖 `execute_model_step()` 的返回值传递异步 Get completion，而是为每个 worker 增加独立的跨进程 completion 通道。**

## 推荐架构

```text
LLMEngine / driver
  ├─ Scheduler
  ├─ KVCompletionAggregator
  └─ rank completion queues
       ▲
       │
rank0 Worker ── KVPoolWorker ── Store Get thread
rank1 Worker ── KVPoolWorker ── Store Get thread
rank2 Worker ── KVPoolWorker ── Store Get thread
```

每个 worker 的后台线程只负责：

```text
执行 Store Get
写本 rank GPU KV block
发布本 rank completion
```

driver 负责：

```text
读取所有 rank completion
按 request_id/load_id 聚合
全部参与 rank 完成后通知 Scheduler
```

## 为什么不用 worker 返回值

当前 DeepInfer 的 `run_workers()`：

```python
cp_broadcast_object((method, kwargs))
return getattr(self, method)(...)
```

只收集 driver 自己的返回值。从属 worker 的返回值没有回传机制。

如果修改 `run_workers()` 让它同步等待每个 worker 返回，会产生两个问题：

1. 异步 Get completion 不一定发生在当前模型 step 内；
2. 为了等返回值，可能把后台传输重新变成同步调用。

因此 completion 应该独立于模型执行 RPC。

## 推荐的第一版实现

### 1. 每个 worker 一个 outbound completion queue

在创建 worker 前，由 driver 创建：

```text
rank0_completion_queue
rank1_completion_queue
rank2_completion_queue
...
```

然后把对应 queue 句柄传给每个 worker。

每个 worker 初始化：

```python
KVPoolWorker(
    store_client=...,
    gpu_blocks=...,
    completion_queue=rank_completion_queue,
    rank=global_rank,
)
```

后台 Get 完成后：

```python
completion_queue.put(
    KVLoadCompletion(
        request_id=...,
        load_id=...,
        rank=...,
        success=...,
        block_results=...,
    )
)
```

建议使用“每个 rank 一个 queue”，而不是所有 rank 共用一个 MPSC queue：

```text
driver poll rank0 queue
driver poll rank1 queue
driver poll rank2 queue
```

这样更容易定位：

- 哪个 rank 没有完成；
- 哪个 rank 失败；
- 哪个 rank 队列积压；
- 是否有重复 completion。

这个 queue 只传递小型控制消息，不传 GPU tensor。

### 2. 任务元数据通过现有广播路径传递

不能把完整 `scheduler_outputs` 直接依赖从属 worker 接收，因为当前 `driver_kwargs` 只给 driver。

应从 `scheduler_outputs` 中提取可序列化的任务：

```python
kv_load_tasks = scheduler_outputs.kv_load_tasks
```

然后作为普通 `kwargs` 广播：

```python
_run_workers(
    "execute_model_step",
    driver_kwargs={
        "seq_metadata_list": seq_metadata_list,
        "scheduler_outputs": scheduler_outputs,
    },
    is_prompt=scheduler_outputs.prompt_run,
    is_placeholder=scheduler_outputs.is_empty(),
    kv_load_tasks=kv_load_tasks,
)
```

这样：

```text
driver Worker:
  有 scheduler_outputs + kv_load_tasks

从属 Worker:
  有 kv_load_tasks
  没有 scheduler_outputs
```

每个 rank 由自己的 worker connector：

```text
根据 task + 本 rank 信息生成 key
绑定本 rank gpu_blocks
提交本 rank Store Get
```

### 3. driver 每轮开始时聚合 completion

在 `LLMEngine.step()` 开头：

```python
completions = self.kv_completion_aggregator.poll()
ready = self.kv_completion_aggregator.apply(completions)
self.scheduler.apply_kv_completions(ready)
```

聚合器维护：

```text
(request_id, load_id)
  -> expected ranks
  -> completed ranks
  -> failed ranks
```

例如：

```text
expected = {0, 1, 2, 3}

收到 rank0 -> {0}
收到 rank1 -> {0, 1}
收到 rank2 -> {0, 1, 2}
收到 rank3 -> {0, 1, 2, 3}
=> load 完成
```

只有全部参与 rank 完成后，才把请求从：

```text
kv_loading -> kv_ready
```

失败则向 Scheduler 报告整体失败。

## 需要明确的三个对象

### Scheduler-side connector

位于 driver/Engine 侧，负责：

```text
构造 KVLoadTask
记录 load_id
记录参与 rank
轮询 completion
聚合 completion
通知 C++ Scheduler
```

### Worker-side connector

每个 rank 一个，位于对应 Worker 进程，负责：

```text
本地 Store client
本地 GPU cache
本地 request queue
本地 Get/Put 线程
向 driver 发布 completion
```

### Completion aggregator

只需要 driver 侧一份，负责：

```text
按 request_id/load_id 聚合 rank 状态
```

## completion 必须携带的字段

不能只传 `request_id`，建议至少：

```text
request_id
load_id
rank
completion_group
success
block_results
error
```

其中：

- `request_id`：对应请求；
- `load_id`：区分同一请求的多次 Get；
- `rank`：确认哪个 rank 完成；
- `completion_group`：区分 TP/CP/DP 聚合范围；
- `block_results`：处理部分 block 失败；
- `error`：保留失败原因。

## 取消和旧任务

请求取消后，后台 Get 可能仍在写目标 block，因此：

```text
abort request
  -> 标记 load_id 已取消
  -> 暂不释放目标 block
  -> 等待对应 completion
  -> 丢弃旧 completion
  -> 释放 block
```

如果没有 `load_id`，旧 Get 可能误更新重试后的新任务。

## 推荐实施顺序

第一版建议先做：

```text
每 rank 一个 worker connector
每 rank 一个 outbound completion queue
driver-side completion aggregator
Get 成功/失败两种结果
暂不做复杂重试
```

先验证这条链路：

```text
Scheduler
  -> KVLoadTask
  -> 广播到所有 worker
  -> 每 rank 异步 Get
  -> 每 rank completion queue
  -> driver 聚合
  -> Scheduler.apply_kv_completions()
```

这套方案不要求修改 `run_workers()` 的返回语义，也不要求从属 worker 返回 `request_outputs`，因此和当前 DeepInfer 的进程模型兼容性最好。
