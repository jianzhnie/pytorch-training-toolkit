#  Torch DDP

Distributed and Parallel Training for PyTorch

## 概述

- 随着基础模型的流行，对高效学习大型模型的需求日益增加。
- 本文将介绍：
  - PyTorch中分布式学习的基本原理
  - 有哪些分布式学习方法
  - 如何区分使用不同的分布式学习方法

### **通信方式**

torch.distributed 的底层通信主要使用 Collective Communication (c10d) library 来支持跨组内的进程发送张量，并主要支持两种类型的通信 API：

- collective communication APIs: 
  - Distributed Data-Parallel Training (DDP)
- P2P communication APIs: 
  - RPC-Based Distributed Training (RPC)

这两种通信 API 在 PyTorch 中分别对应了两种分布式训练方式：Distributed Data-Parallel Training (DDP) 和 RPC-Based Distributed Training (RPC)。本文着重探讨 Distributed Data-Parallel Training (DDP) 的通信方式和 API。

## **torch.distributed 概念与定义**

[分布式数据并行 (DDP)](https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel) 是 PyTorch 中一个强大的模块，允许在多个机器上并行化你的模型，非常适合大规模深度学习应用。

**定义**：首先我们提供 Torch.distributed 的官方定义

- `torch.distributed` 包为运行在一台或多台机器上的多个计算节点之间的 **PyTorch 提供支持多进程并行性通信的原语**， 能轻松地将跨进程和机器集群的计算并行化。

- `torch.nn.parallel.DistributedDataParallel (DDP)` 是建立在此功能的基础上，为任何PyTorch模型提供同步分布式训练的包装器。

可以注意到的是，torch.distributed 的核心功能是进行多进程级别的通信（而非多线程），以此达到多卡多机分布式训练的目的。这与 `Multiprocessing` 包 - `torch.multiprocessing` 和 `torch.nn.DataParallel()` 提供的并行性类型不同，因为它支持多台网络连接的机器，并且用户必须为每个进程显式启动主训练脚本的单独副本。

### `DataParallel` 和 `DistributedDataParallel` 的比较

在我们深入之前，让我们澄清一下为什么你需要考虑使用 `DistributedDataParallel` 而不是 `DataParallel`，尽管它的复杂性增加了：

- 首先，`DataParallel` 是单进程、多线程的，但它只能在单个机器上工作。相比之下，`DistributedDataParallel` 是多进程的，支持单机和多机训练。
- 每个进程包含一个独立的Python解释器，消除了从单个Python进程驱动多个执行线程、模型副本或GPUs带来的额外解释器开销和“GIL-thrashing”。这对于大量使用Python运行时的模型尤其重要，包括具有递归层或许多小组件的模型。由于线程间的 GIL 争用、每次迭代的复制模型以及分散输入和收集输出的额外开销，`DataParallel` 通常比 `DistributedDataParallel` 更慢，即使在单个机器上也是如此。
- 每个进程维护自己的优化器，并在每次迭代中执行完整的优化步骤。虽然这看起来是多余的，因为梯度已经在进程间聚集并平均化，因此对于每个进程来说都是相同的，但这意味着不需要参数广播步骤，减少了在节点间传输张量所花费的时间。
- 如果你的模型太大而无法放入单个 GPU，你必须使用**模型并行**将其拆分到多个 GPU 上。`DistributedDataParallel` 与**模型并行**一起工作，而 `DataParallel` 目前不支持。当 DDP 与模型并行结合时，每个 DDP 进程将使用模型并行，所有进程共同使用数据并行。

| 特性         | DataParallel                                              | DistributedDataParallel      |
| ------------ | --------------------------------------------------------- | ---------------------------- |
| 开销         | 更多开销；模型在每次前向传播时都会被复制和销毁            | 模型只复制一次               |
| 支持的并行性 | 仅支持单节点并行                                          | 支持扩展到多台机器           |
| 速度         | 较慢；使用单进程中的多线程，并且会遇到全局解释器锁（GIL） | 更快（没有 GIL），使用多进程 |

## torch.distributed 通信后端

`torch.distributed `支持三个内置后端 `GLOO, MPI, NCCL`，每个后端都有不同的功能。下表显示了哪些函数可用于 CPU / CUDA 张量。

| Backend        | `gloo` | `gloo` | `mpi` | `mpi` | `nccl` | `nccl` |
| -------------- | :----: | :----: | :---: | :---: | :----: | :----: |
| Device         |  CPU   |  GPU   |  CPU  |  GPU  |  CPU   |  GPU   |
| send           |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| recv           |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| broadcast      |   ✓    |   ✓    |   ✓   |   ?   |   ✘    |   ✓    |
| all_reduce     |   ✓    |   ✓    |   ✓   |   ?   |   ✘    |   ✓    |
| reduce         |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| all_gather     |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| gather         |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| scatter        |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| reduce_scatter |   ✘    |   ✘    |   ✘   |   ✘   |   ✘    |   ✓    |
| all_to_all     |   ✘    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| barrier        |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |

默认情况下，对于 Linux，Gloo 和 NCCL 后端已构建并包含在 PyTorch 分布式版中（仅在使用 CUDA 构建时才包含 NCCL）。MPI 是一个可选后端，只有从源代码构建 PyTorch 时才可以包含它。（例如，在安装了 MPI 的主机上构建 PyTorch。）

#### 使用哪个后端？

过去，我们经常被问到：“我应该使用哪个后端？”。

- 经验法则
  - 对于分布式 **GPU** 训练，使用 NCCL 后端
  - 对于分布式 **CPU** 训练，使用 Gloo 后端。
- 具有 InfiniBand 互连的 GPU 主机
  - 使用 NCCL，因为它是唯一目前支持 InfiniBand 和 GPUDirect 的后端。
- 具有以太网互连的 GPU 主机
  - 使用 NCCL，因为它目前提供了最佳的分布式 GPU 训练性能，特别是对于多进程单节点或多节点分布式训练。如果您遇到任何问题，请使用 Gloo 作为回退选项。（注意，Gloo 目前对于 GPU 的运行速度比 NCCL 慢。）
- 具有 InfiniBand 互连的 CPU 主机
  - 如果您的 InfiniBand 启用了 IP over IB，使用 Gloo，否则使用 MPI。我们计划在即将发布的版本中为 Gloo 添加 InfiniBand 支持。
- 具有以太网互连的 CPU 主机
  - 使用 Gloo，除非有特定原因使用 MPI。

## 初始化分布式进程

分布式包在使用之前，需要通过 [`torch.distributed.init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group) 或 [`torch.distributed.device_mesh.init_device_mesh()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.device_mesh.init_device_mesh) 函数进行初始化。两者都会阻塞，直到所有进程加入。

> 警告
>
> 初始化不是线程安全的。进程组创建应从单个线程执行，以防止在不同rank上分配不一致的“UUID”，并防止在初始化期间导致竞争条件，从而导致挂起。

有几个关键函数：

### `distributed.is_available()`

- 如果分布式包可用，返回 `True`。否则，`torch.distributed` 不会暴露任何其他 API。

### `distributed.init_process_group`

- 初始化默认的分布式进程组。这还将初始化分布式包。

- 有两种主要方式来初始化进程组：

  - 显式指定 `store`、`rank` 和 `world_size`。

  - 指定 `init_method`（一个 URL 字符串），指示如何/在哪里发现对等方。可以选择指定 `rank` 和 `world_size`，或者将所有必需参数编码在 URL 中并省略它们。

- 如果没有指定，则假定 `init_method` 为 `“env://”`。

核心参数: 

- **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*Backend*](https://pytorch.org/docs/main/distributed.html#torch.distributed.Backend)*,* *optional*) – 要使用的后端。根据构建时配置，有效值包括 `mpi`、`gloo`、`nccl` 和 `ucc`。如果未提供后端，则将创建 `gloo` 和 `nccl` 后端.
- **init_method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – 指定如何初始化进程组的 URL。如果未指定 `init_method` 或 `store`，则默认为“env://”。与 `store` 互斥。
- **world_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 参与作业的进程数。如果指定了 `store`，则为必需。
- **rank** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 当前进程的rank（它应该是一个介于 0 和 `world_size`-1 之间的数字）。如果指定了 `store`，则为必需。
- **store** ([*Store*](https://pytorch.org/docs/main/distributed.html#torch.distributed.Store)*,* *optional*) – 所有进程可访问的键/值存储，用于交换连接/地址信息。与 `init_method` 互斥。
- **timeout** (*timedelta**,* *optional*) – 针对进程组执行操作的超时时间。默认值为 NCCL 为 10 分钟，其他后端为 30 分钟。这是在异步取消集合运算并使进程崩溃后的持续时间。这样做是因为 CUDA 执行是异步的，继续执行用户代码不再安全，因为失败的异步 NCCL 操作可能会导致后续 CUDA 操作在损坏的数据上运行。当设置 TORCH_NCCL_BLOCKING_WAIT 时，进程将阻塞并等待此超时。

### `distributed.is_initialized()`

- 检查默认进程组是否已初始化。返回类型 [bool](https://docs.python.org/3/library/functions.html#bool)

### `distributed.is_mpi_available()`

- 检查 MPI 后端是否可用。返回类型[bool](https://docs.python.org/3/library/functions.html#bool)

### `distributed.is_gloo_available()`

- 检查 Gloo 后端是否可用。返回类型[bool](https://docs.python.org/3/library/functions.html#bool)

### `distributed.is_torchelastic_launched()`

检查此进程是否由 `torch.distributed.elastic`（又名 torchelastic）启动。使用 `TORCHELASTIC_RUN_ID` 环境变量的存在作为代理来确定当前进程是否由 torchelastic 启动。这对于作业 ID 用于对等发现目的的非空值是合理的。返回类型[bool](https://docs.python.org/3/library/functions.html#bool)

### 初始化方法

目前支持三种初始化方法：

#### TCP 初始化

有两种方式使用 TCP 初始化，都需要一个所有进程都可访问的网络地址和一个所需的 `world_size`。第一种方式需要指定属于rank 0 进程的地址。这种初始化方法要求所有进程手动指定rank。

```python
import torch.distributed as dist

# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                        rank=args.rank, world_size=4)
```

#### 共享文件系统初始化

另一种初始化方法利用共享文件系统，该文件系统对组中的所有机器可见，并带有所需的 `world_size`。URL 应以 `file://` 开头，并包含共享文件系统上现有目录中的路径（在现有目录中）。文件系统初始化将自动创建该文件（如果它不存在），但不会删除该文件。因此，您有责任确保在下次在同一文件路径/名称上调用 [`init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group) 之前清理该文件。

> 警告
>
> 此方法将始终创建文件并尝试在程序结束时清理并删除该文件。换句话说，每次使用文件初始化方法时，都需要一个新的空文件，以便初始化成功。如果使用前一次初始化（未清理）的同一文件再次调用，这是意外行为，并且通常会导致死锁和失败。因此，即使此方法会尽力清理文件，如果自动删除不成功，您有责任确保在训练结束时删除该文件，以防止在下次调用 [`init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group) 时再次使用该文件。这在计划多次调用 [`init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group) 时尤其重要。换句话说，如果文件未删除/清理，并且您再次调用 [`init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group)，则预期会发生失败。我们的经验法则是，确保每次调用 [`init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group) 时，文件不存在或为空。

```python
import torch.distributed as dist

# rank应始终指定
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)
```

#### 环境变量初始化

此方法将从环境变量中读取配置，允许完全自定义如何获取信息。要设置的变量是：

- `MASTER_PORT` - 必需；rank 0 机器上的空闲端口
- `MASTER_ADDR` - 必需（rank 0 除外）；rank 0 节点的地址
- `WORLD_SIZE` - 必需；可以在此处设置，也可以在调用 init 函数时设置
- `RANK` - 必需；可以在此处设置，也可以在调用 init 函数时设置

rank 0 的机器将用于设置所有连接。

这是默认方法，意味着 `init_method` 不必指定（或可以为 `env://`）。

```python
import torch.distributed as dist

# rank应始终指定
dist.init_process_group(backend, init_method='env://')
```



## 初始化后

一旦运行了 [`torch.distributed.init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group)，以下函数就可以使用了。要检查进程组是否已经初始化，请使用 [`torch.distributed.is_initialized()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.is_initialized)。

- `torch.distributed.Backend(*name*)`
- `torch.distributed.get_backend(*group=None*)`
- `torch.distributed.get_rank(*group=None*)`
- `torch.distributed.get_world_size(*group=None*)`

## Shutdown

在退出时清理资源很重要，通过调用 `destroy_process_group()` 来实现。

最简单的模式是在训练脚本中不再需要通信的点（通常在主函数的末尾附近），对每个训练器进程调用 `destroy_process_group()`，而不是在外部进程启动器级别。

如果 `destroy_process_group()` 没有被所有等级在超时持续时间内调用，尤其是在应用程序中有多个进程组时（例如，用于 N-D 并行），退出时可能会出现挂起。这是因为 ProcessGroupNCCL 的析构函数调用 `ncclCommAbort`，这必须是集合运算调用的，但 Python 的 GC 调用 ProcessGroupNCCL 的析构函数的顺序是不确定的。调用 `destroy_process_group()` 有助于确保 `ncclCommAbort` 以一致的顺序在所有等级上调用，并避免在 ProcessGroupNCCL 的析构函数期间调用 `ncclCommAbort`。

## 点对点通信

- [`send()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.send)
- [`recv()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.recv)
- [`isend()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.isend)
- [`irecv()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.irecv)

最简单的多进程通信方式是点对点通信。信息从一个进程被发送到另一个进程。

### send()、recv()、isend()、irecv()

![img](https://pic1.zhimg.com/v2-bc887111330cd0225c68c7cd353dae0d_720w.jpg?source=d16d100b)

```python
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

在上面的示例中，两个进程都从 tensor(0) 开始，然后进程 0 递增张量并将其发送到进程 1，以便它们都以 tensor(1) 结尾。 请注意，进程 1 需要分配内存以存储它将接收的数据。

另请注意，send / recv 被**阻塞**：两个过程都停止，直到通信完成。我们还有另外一种无阻塞的通信方式，请看下例

```python
"""Non-blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

[`isend()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.isend) 和 [`irecv()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.irecv) 在使用时返回分布式请求对象。通常，这些对象的类型是未指定的，因为它们不应手动创建，但它们保证支持两种方法：

- `is_completed()` - 如果操作已完成，返回 True
- `wait()` - 将阻塞进程，直到操作完成。`is_completed()` 保证在返回时返回 True。

我们通过调用 wait 函数以使自己在子进程执行过程中保持休眠状态。由于我们不知道何时将数据传递给其他进程，因此在 req.wait() 完成之前，我们既不应该修改发送的张量也不应该访问接收的张量以防止不确定的写入.

##  进程组间通信

- [`broadcast()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.broadcast)
- [`reduce()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.reduce)
- [`all_reduce()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.all_reduce)
- [`gather()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.gather)
- [`all_gather()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.all_gather)
- [`scatter()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.scatter)
- [`reduce_scatter()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.reduce_scatter)
- [`ReduceOp`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)

### 集合通信

与点对点通信相反，集合通信允许跨组中所有进程的通信模式。例如，为了获得所有过程中所有张量的总和，我们可以使用 dist.all_reduce(tensor, op, group) 函数进行组间通信

```python
""" All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```

这段代码首先将进程 0 和 1 组成进程组，然后将各自进程中 tensor(1) 相加。由于我们需要组中所有张量的总和，因此我们将 dist.reduce_op.SUM 用作化简运算符。 一般来说，任何可交换的数学运算都可以用作运算符。 PyTorch 开箱即用，带有 4 个这样的运算符，它们都在元素级运行：

- dist.reduce_op.SUM
- dist.reduce_op.PRODUCT
- dist.reduce_op.MAX
- dist.reduce_op.MIN

### `distributed.ReduceOp`

`torch.distributed.ReduceOp` 是一个类似于枚举的类，用于表示可用的归约操作：`SUM`、`PRODUCT`、`MIN`、`MAX`、`BAND`、`BOR`、`BXOR` 和 `PREMUL_SUM`。

在使用 NCCL 后端时，`BAND`、`BOR` 和 `BXOR` 归约操作不可用。

`AVG` 操作会在对值进行求和之前将其除以世界大小（world size）。`AVG` 仅在使用 NCCL 后端时可用，并且仅适用于 NCCL 版本 2.10 或更高版本。

此外，`MAX`、`MIN` 和 `PRODUCT` 操作不支持复数张量。

可以通过属性访问该类的值，例如 `ReduceOp.SUM`。它们用于指定归约集体操作的策略，例如 `reduce()`。

除了 dist.all_reduce(tensor, op, group) 之外，PyTorch 中目前共有 6 种组间通信方式

### distributed.scatter

distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)： 将张量 scatter_list[i] 复制第 i 个进程的过程。 例如，在实现分布式训练时，我们将数据分成四份并分别发送到不同的机子上计算梯度。scatter 函数可以用来将信息从 src 进程发送到其他进程上。

<img src="https://pic1.zhimg.com/v2-812552c20c0785cf5dbd1f2182e79b9d_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

| tensor       | 发送的数据                                  |
| ------------ | ------------------------------------------- |
| scatter_list | 存储发送数据的列表（只需在 src 进程中指定） |
| dst          | 发送进程的rank                              |
| group        | 指定进程组                                  |
| async_op     | 该 op 是否是异步操作                        |

### distributed.gather

distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)： 从 dst 中的所有进程复制 tensor。例如，在实现分布式训练时，不同进程计算得到的梯度需要汇总到一个进程，并计算平均值以获得统一的梯度。gather 函数可以将信息从别的进程汇总到 dst 进程。

<img src="https://pic1.zhimg.com/v2-602a2ed1126c9ef56e53235ab3f8adeb_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

|             |                                           |
| ----------- | ----------------------------------------- |
| tensor      | 接受的数据                                |
| gather_list | 存储接受数据的列表（只需在dst进程中指定） |
| dst         | 汇总进程的rank                            |
| group       | 指定进程组                                |
| async_op    | 该op是否是异步操作                        |

### distributed.reduce

distributed.reduce(tensor, dst, op, group)：将 op 应用于所有 tensor，并将结果存储在 dst 中。

<img src="https://pic1.zhimg.com/v2-348e954c4c77ef281c6204bccf0c8f5f_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />



### distributed.all_reduce

distributed.all_reduce(tensor, op, group)： 与 reduce 相同，但是结果存储在所有进程中。

<img src="https://picx.zhimg.com/v2-b7597d4d57bbc6ba47a59166b8331d8f_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />



### distributed.broadcast

distributed.broadcast(tensor, src, group)：将张量广播给整个组。将tensor从src复制到所有其他进程。

`tensor`参与集体的所有进程必须具有相同数量的元素。





<img src="https://pic1.zhimg.com/v2-8ae0a62a27420e1de19fbbea5dc9a09b_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

### distributed.all_gather

distributed.all_gather(tensor_list, tensor, group)：将所有进程中的 tensor 从所有进程复制到 tensor_list

<img src="https://picx.zhimg.com/v2-21ce7cb6b3be25d25ee02b5fe0b9c70c_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

## 同步和异步集合运算

每个集合运算函数都支持以下两种操作，具体取决于传递给集合运算的 `async_op` 标志的设置：

**同步操作** - 默认模式，当 `async_op` 设置为 `False` 时。函数返回时，保证集合运算已完成。在 CUDA 操作的情况下，不保证 CUDA 操作已完成，因为 CUDA 操作是异步的。对于 CPU 集合运算，任何进一步的函数调用使用集合运算调用的输出将按预期运行。对于 CUDA 集合运算，在同一 CUDA 流上的函数调用使用输出将按预期运行。用户必须注意不同流下的同步情况。有关 CUDA 语义的详细信息，如流同步，请参阅 [CUDA 语义](https://pytorch.org/docs/stable/notes/cuda.html)。请参阅以下脚本，了解 CPU 和 CUDA 操作在这些语义上的差异示例。

**异步操作** - 当 `async_op` 设置为 `True` 时。集合运算函数返回一个分布式请求对象。通常，你不需要手动创建它，并且它保证支持两种方法：

* `is_completed()` - 对于 CPU 集合运算，如果操作已完成，则返回 `True`。对于 CUDA 操作，如果操作已成功排队到 CUDA 流，并且输出可以在默认流上使用而无需进一步同步，则返回 `True`。
* `wait()` - 对于 CPU 集合运算，将阻塞进程直到操作完成。对于 CUDA 集合运算，将阻塞直到操作已成功排队到 CUDA 流，并且输出可以在默认流上使用而无需进一步同步。
* `get_future()` - 返回 `torch._C.Future` 对象。支持 NCCL，也支持 GLOO 和 MPI 上的大多数操作，除了点对点操作。注意：随着我们继续采用 Futures 并合并 API，`get_future()` 调用可能会变得冗余。

**示例**

以下代码可以作为在使用分布式集合运算时 CUDA 操作语义的参考。它展示了在使用不同 CUDA 流上的集合运算输出时显式同步的需求：

```python
# 代码在每个 rank 上运行。
dist.init_process_group("nccl", rank=rank, world_size=2)
output = torch.tensor([rank]).cuda(rank)
s = torch.cuda.Stream()
handle = dist.all_reduce(output, async_op=True)
# Wait 确保操作已排队，但不保证完成。
handle.wait()
# 在非默认流上使用结果。
with torch.cuda.stream(s):
    s.wait_stream(torch.cuda.default_stream())
    output.add_(100)
if rank == 0:
    # 如果省略显式调用 wait_stream，下面的输出将不确定地为 1 或 101，
    # 具体取决于 allreduce 是否在 add 完成之前覆盖了值。
    print(output)
```

## 分布式程序启动工具

`torch.distributed` 包还提供了一个启动工具 `torch.distributed.launch`。这个辅助工具可以用于在每个训练节点上启动多个进程进行分布式训练。

`torch.distributed.launch` 是一个模块，它会在每个训练节点上生成多个分布式训练进程。

> 警告
>
> 该模块将被弃用，取而代之的是 `torchrun`。

该工具可用于单节点分布式训练，其中每个节点将生成一个或多个进程。该工具可用于 CPU 训练或 GPU 训练。如果该工具用于 GPU 训练，每个分布式进程将在单个 GPU 上运行。这可以显著提高单节点训练性能。它还可以用于多节点分布式训练，通过在每个节点上生成多个进程来提高多节点分布式训练性能。这对于具有多个支持直接 GPU 的 Infiniband 接口的系统尤其有益，因为所有这些接口都可以用于聚合通信带宽。

在单节点分布式训练或多节点分布式训练的情况下，该工具将启动每个节点的给定数量的进程（`--nproc-per-node`）。如果用于 GPU 训练，这个数量需要小于或等于当前系统上的 GPU 数量（`nproc_per_node`），并且每个进程将在从 GPU 0 到 GPU (`nproc_per_node - 1`) 的单个 GPU 上运行。

如何使用该模块：

单节点多进程分布式训练

```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 以及你的训练脚本的所有其他参数)
```

多节点多进程分布式训练（例如，两个节点）

节点 1：（IP: 192.168.1.1，有一个空闲端口：1234）

```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
           --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           以及你的训练脚本的所有其他参数)
```

节点 2：

```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node-rank=1 --master-addr="192.168.1.1"
           --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           以及你的训练脚本的所有其他参数)
```

要查看该模块提供的可选参数：

```bash
python -m torch.distributed.launch --help
```

重要提示：

1. 该工具和多进程分布式（单节点或多节点）GPU 训练目前仅在使用 NCCL 分布式后端时才能达到最佳性能。因此，NCCL 后端是用于 GPU 训练的推荐后端。

2. 在你的训练程序中，你必须解析命令行参数 `--local-rank=LOCAL_PROCESS_RANK`，该参数将由该模块提供。如果你的训练程序使用 GPU，你应该确保你的代码仅在 `LOCAL_PROCESS_RANK` 对应的 GPU 设备上运行。可以通过以下方式实现：

解析 `local_rank` 参数

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int)
args = parser.parse_args()
```

将设备设置为本地排名，使用以下任一方式：

```python
torch.cuda.set_device(args.local_rank)  # 在你的代码运行之前
```

或

```python
with torch.cuda.device(args.local_rank):
    # 你的代码运行
    ...
```

在版本 2.0.0 中更改：启动器会将 `--local-rank=<rank>` 参数传递给你的脚本。从 PyTorch 2.0.0 开始，破折号 `--local-rank` 是首选，而不是之前使用的下划线 `--local_rank`。

为了向后兼容，用户可能需要在参数解析代码中处理这两种情况。这意味着在参数解析器中同时包含 `--local-rank` 和 `--local_rank`。如果只提供 `--local-rank`，启动器将触发错误：“error: unrecognized arguments: –local-rank=<rank>”。对于仅支持 PyTorch 2.0.0+ 的训练代码，包含 `--local-rank` 应该足够。

3. 在你的训练程序中，你应该在开始时调用以下函数来启动分布式后端。强烈建议使用 `init_method=env://`。其他初始化方法（例如 `tcp://`）可能有效，但 `env://` 是该模块官方支持的方法。

```python
torch.distributed.init_process_group(backend='YOUR BACKEND',
                                     init_method='env://')
```

4. 在你的训练程序中，你可以使用常规的分布式函数或使用 `torch.nn.parallel.DistributedDataParallel()` 模块。如果你的训练程序使用 GPU 进行训练并希望使用 `torch.nn.parallel.DistributedDataParallel()` 模块，可以按如下方式配置：

```python
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)
```

请确保 `device_ids` 参数设置为你的代码将运行的唯一 GPU 设备 ID。这通常是进程的本地排名。换句话说，`device_ids` 需要是 `[args.local_rank]`，`output_device` 需要是 `args.local_rank`，以便使用该工具。

5. 另一种通过环境变量 `LOCAL_RANK` 将 `local_rank` 传递给子进程的方式。当你使用 `--use-env=True` 启动脚本时，此行为将被启用。你必须调整上述子进程示例，将 `args.local_rank` 替换为 `os.environ['LOCAL_RANK']`；当你指定此标志时，启动器不会传递 `--local-rank`。

警告

`local_rank` 不是全局唯一的：它在机器上的每个进程中是唯一的。因此，不要使用它来决定是否应该执行某些操作，例如写入网络文件系统。参见 https://github.com/pytorch/pytorch/issues/12042 了解如果不正确处理可能会出现的问题。

## 生成工具

`torch.multiprocessing` 包还提供了一个生成函数 `torch.multiprocessing.spawn()`。这个辅助函数可以用于生成多个进程。它通过传入你想要运行的函数并生成 N 个进程来运行它。这也可以用于多进程分布式训练。

有关如何使用它的参考，请参阅 PyTorch 示例 - ImageNet 实现。

请注意，此函数需要 Python 3.4 或更高版本。