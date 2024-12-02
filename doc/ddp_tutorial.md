# 分布式数据并行入门

## 先决条件

- [PyTorch 分布式概述](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DistributedDataParallel API 文档](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)

[分布式数据并行 (DDP)](https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel) 是 PyTorch 中一个强大的模块，允许你在多个机器上并行化你的模型，非常适合大规模深度学习应用。要使用 DDP，你需要生成多个进程并在每个进程中创建一个 DDP 实例。

但它是如何工作的呢？DDP 使用来自 [torch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html) 包的集体通信来同步梯度和缓冲区。这意味着每个进程都会有模型的一个副本，但它们会一起工作，就像模型在单个机器上一样进行训练。

为了实现这一点，DDP 为模型中的每个参数注册了一个 autograd 钩子。当运行反向传播时，这个钩子会触发并在所有进程之间同步梯度。这确保每个进程都有相同的梯度，然后用于更新模型。

使用 DDP 的推荐方式是为每个模型副本生成一个进程。模型副本可以跨多个设备。DDP 进程可以放在同一台机器上或跨机器。请注意，GPU 设备不能在 DDP 进程之间共享（即一个 GPU 对应一个 DDP 进程）。

在本教程中，我们将从一个基本的 DDP 用例开始，然后演示更多高级用例，包括模型检查点和将 DDP 与模型并行结合。

**注意**: 本教程中的代码在一个 8-GPU 服务器上运行，但可以轻松推广到其他环境。

## `DataParallel` 和 `DistributedDataParallel` 的比较

在我们深入之前，让我们澄清一下为什么你会考虑使用 `DistributedDataParallel` 而不是 `DataParallel`，尽管它的复杂性增加了：

- 首先，`DataParallel` 是单进程、多线程的，但它只能在单个机器上工作。相比之下，`DistributedDataParallel` 是多进程的，支持单机和多机训练。由于线程间的 GIL 争用、每次迭代的复制模型以及分散输入和收集输出的额外开销，`DataParallel` 通常比 `DistributedDataParallel` 更慢，即使在单个机器上也是如此。
- 回顾 [之前的教程](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)，如果你的模型太大而无法放入单个 GPU，你必须使用**模型并行**将其拆分到多个 GPU 上。`DistributedDataParallel` 与**模型并行**一起工作，而 `DataParallel` 目前不支持。当 DDP 与模型并行结合时，每个 DDP 进程将使用模型并行，所有进程共同使用数据并行。

## 基本用例

要创建一个 DDP 模块，你必须首先正确设置进程组。更多细节可以在 [使用 PyTorch 编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html) 中找到。

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# 在 Windows 平台上，torch.distributed 包仅支持 Gloo 后端、FileStore 和 TcpStore。
# 对于 FileStore，在 init_process_group 中设置 init_method 参数为本地文件。示例如下：
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# 对于 TcpStore，与在 Linux 上的方式相同。

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

现在，让我们创建一个 Toy 模块，用 DDP 包装它，并给它一些虚拟输入数据。请注意，由于 DDP 在构造函数中将模型状态从 rank 0 进程广播到所有其他进程，因此你不需要担心不同的 DDP 进程从不同的初始模型参数值开始。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # 创建模型并将其移动到带有 id rank 的 GPU 上
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

正如你所见，DDP 包装了较低级别的分布式通信细节，并提供了一个干净的 API，就像它是一个本地模型一样。梯度同步通信在反向传播期间进行，并与反向计算重叠。当 `backward()` 返回时，`param.grad` 已经包含了同步的梯度张量。对于基本用例，DDP 只需要几行额外的代码来设置进程组。当将 DDP 应用于更高级的用例时，一些注意事项需要谨慎。

## 处理速度不均衡

在 DDP 中，构造函数、前向传播和反向传播是分布式同步点。不同的进程预计会启动相同数量的同步，并以相同的顺序到达这些同步点，并且大致同时进入每个同步点。否则，较快的进程可能会提前到达并在等待掉队者时超时。因此，用户负责平衡跨进程的工作负载分布。有时，由于网络延迟、资源争用或不可预测的工作负载峰值等原因，处理速度不均衡是不可避免的。为了避免在这些情况下超时，请确保在调用 [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 时传递一个足够大的 `timeout` 值。

## 保存和加载 `checkppoint`

在训练期间使用 `torch.save` 和 `torch.load` 检查点模块并从中恢复是很常见的。有关更多详细信息，请参阅 [保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)。在使用 DDP 时，一个优化是在一个进程中保存模型，然后在所有进程中加载它，以减少写入开销。这之所以有效，是因为所有进程都从相同的参数开始，并且在反向传播中同步梯度，因此优化器应该保持将参数设置为相同的值。如果你使用这种优化（即在一个进程中保存但在所有进程中恢复），请确保在保存完成之前没有进程开始加载。此外，在加载模块时，你需要提供一个适当的 `map_location` 参数，以防止进程进入其他设备。如果缺少 `map_location`，`torch.load` 会首先将模块加载到 CPU，然后将每个参数复制到保存它的地方，这会导致同一台机器上的所有进程使用相同的一组设备。有关更高级的故障恢复和弹性支持，请参阅 [TorchElastic](https://pytorch.org/elastic)。

```python
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # 所有进程应该看到相同的参数，因为它们都从相同的随机参数开始，并且在反向传播中同步梯度。
        # 因此，在一个进程中保存它是足够的。
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # 使用 barrier() 确保进程 1 在进程 0 保存模型后加载模型。
    dist.barrier()
    # 配置 map_location 正确
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # 不需要使用 dist.barrier() 来保护下面的文件删除操作
    # 因为 DDP 中的 AllReduce 操作已经在反向传播中起到了同步作用。

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")
```

## 将 DDP 与模型并行结合

DDP 也适用于多 GPU 模型。当训练具有大量数据的庞大模型时，DDP 包装多 GPU 模型尤其有用。

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

当将多 GPU 模型传递给 DDP 时，`device_ids` 和 `output_device` 必须**不**设置。输入和输出数据将由应用程序或模型 `forward()` 方法放置在适当的设备上。

```python
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # 为这个进程设置 mp_model 和设备
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs 将在 dev1 上
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running DDP with model parallel example on rank {rank}.")

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    world_size = n_gpus//2
    run_demo(demo_model_parallel, world_size)
```

## 使用 torch.distributed.run/torchrun 初始化 DDP

我们可以利用 PyTorch Elastic 来简化 DDP 代码并更轻松地初始化作业。让我们仍然使用 Toymodel 示例并创建一个名为 `elastic_ddp.py` 的文件。

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # 创建模型并将其移动到带有 id rank 的 GPU 上
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    demo_basic()
```

然后可以在所有节点上运行 `torch elastic/torchrun` 命令来初始化上面创建的 DDP 作业：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
```

在上面的示例中，我们在两台主机上运行 DDP 脚本，每台主机上运行 8 个进程。也就是说，我们在 16 个 GPU 上运行这个作业。请注意，`$MASTER_ADDR` 必须在所有节点上相同。

这里 `torchrun` 将在启动它的节点上生成 8 个进程，并在每个进程上调用 `elastic_ddp.py`，但用户还需要应用集群管理工具（如 slurm）在 2 个节点上实际运行此命令。

例如，在启用了 SLURM 的集群上，我们可以编写一个脚本来运行上面的命令并设置 `MASTER_ADDR` 为：

```bash
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

然后我们可以使用 SLURM 命令 `srun --nodes=2 ./torchrun_script.sh` 运行这个脚本。

这只是一个示例；你可以选择自己的集群调度工具来启动 `torchrun` 作业。

有关 Elastic run 的更多信息，请参阅 [快速入门文档](https://pytorch.org/docs/stable/elastic/quickstart.html)。
