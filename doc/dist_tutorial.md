# 使用PyTorch编写分布式应用程序

在这个简短的教程中，我们将介绍PyTorch的分布式包。我们将了解如何设置分布式环境，使用不同的通信策略，并深入了解该包的一些内部机制。

## 设置

PyTorch中包含的分布式包（即`torch.distributed`）使研究人员和实践者能够轻松地将计算并行化到进程和机器集群中。为此，它利用消息传递语义，允许每个进程将数据传递给任何其他进程。与多进程（`torch.multiprocessing`）包不同，进程可以使用不同的通信后端，并且不限于在同一台机器上执行。

为了开始，我们需要能够同时运行多个进程。如果你有权访问计算集群，你应该咨询本地系统管理员或使用你喜欢的协调工具（例如，[pdsh](https://linux.die.net/man/1/pdsh)，[clustershell](https://cea-hpc.github.io/clustershell/)，或 [slurm](https://slurm.schedmd.com/)。出于本教程的目的，我们将使用单台机器并生成多个进程，使用以下模板。

```python
"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ 稍后实现的分布式函数。 """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ 初始化分布式环境。 """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

上面的脚本生成了两个进程，每个进程都会设置分布式环境，初始化进程组（`dist.init_process_group`），最后执行给定的`run`函数。

让我们看一下`init_process`函数。它确保每个进程都能够通过主进程使用相同的IP地址和端口进行协调。请注意，我们使用了`gloo`后端，但还有其他可用的后端。（参见[第5.1节](https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends)）我们将在本教程的最后介绍`dist.init_process_group`中发生的魔法，但它本质上允许进程通过共享它们的位置相互通信。

## 点对点通信

![Send and Recv](https://pytorch.org/tutorials/_images/send_recv.png)

从一个进程到另一个进程的数据传输称为点对点通信。这些通过`send`和`recv`函数或它们的*立即*对应部分`isend`和`irecv`实现。

```python
"""阻塞点对点通信。"""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # 将张量发送到进程1
        dist.send(tensor=tensor, dst=1)
    else:
        # 从进程0接收张量
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

在上面的示例中，两个进程都从一个零张量开始，然后进程0递增张量并将其发送到进程1，以便它们最终都得到1.0。请注意，进程1需要分配内存以存储它将接收的数据。

还要注意`send/recv`是**阻塞**的：两个进程都会阻塞，直到通信完成。另一方面，立即通信是**非阻塞**的；脚本继续执行，方法返回一个`Work`对象，我们可以选择在其上`wait()`。

```python
"""非阻塞点对点通信。"""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # 将张量发送到进程1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # 从进程0接收张量
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

在使用立即通信时，我们必须小心如何使用发送和接收的张量。因为我们不知道数据何时会被传递到另一个进程，所以在`req.wait()`完成之前，我们不应该修改发送的张量或访问接收的张量。换句话说，

- 在`dist.isend()`之后写入`tensor`将导致未定义的行为。
- 在`dist.irecv()`之后读取`tensor`将导致未定义的行为。

然而，在执行了`req.wait()`之后，我们可以保证通信已经发生，并且存储在`tensor[0]`中的值是1.0。

点对点通信在我们想要更精细地控制进程之间的通信时非常有用。它们可以用于实现花哨的算法，例如[Baidu的DeepSpeech](https://github.com/baidu-research/baidu-allreduce) 或[Facebook的大规模实验](https://research.fb.com/publications/imagenet1kin1h/)。参见[第4.1节](https://pytorch.org/tutorials/intermediate/dist_tuto.html#our-own-ring-allreduce)

## 集合通信

| ![Scatter](https://pytorch.org/tutorials/_images/scatter.png) | ![Gather](https://pytorch.org/tutorials/_images/gather.png) |
| ------------------------------------------------------------- | ----------------------------------------------------------- |
| Scatter                                                       | Gather                                                      |

______________________________________________________________________

| ![Reduce](https://pytorch.org/tutorials/_images/reduce.png) | ![All-Reduce](https://pytorch.org/tutorials/_images/all_reduce.png) |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| Reduce                                                      | All-Reduce                                                          |

______________________________________________________________________

| ![Broadcast](https://pytorch.org/tutorials/_images/broadcast.png) | ![All-Gather](https://pytorch.org/tutorials/_images/all_gather.png) |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| Broadcast                                                         | All-Gather                                                          |

与点对点通信相反，集合通信允许在**组**中的所有进程之间进行通信。组是我们所有进程的一个子集。要创建一个组，我们可以传递一个秩列表给`dist.new_group(group)`。默认情况下，在所有进程上执行通信。例如，为了获得所有进程上张量的总和，我们可以使用`dist.all_reduce(tensor, op, group)` 通信模块。

```python
""" All-Reduce 示例。"""
def run(rank, size):
    """ 简单的集合通信。 """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```

因为我们想要组中所有张量的总和，所以我们使用`dist.ReduceOp.SUM`作为减少操作符。一般来说，任何可交换的数学运算都可以用作操作符。开箱即用，PyTorch带有许多这样的操作符，所有这些操作符都在元素级别工作：

- `dist.ReduceOp.SUM`，
- `dist.ReduceOp.PRODUCT`，
- `dist.ReduceOp.MAX`，
- `dist.ReduceOp.MIN`，
- `dist.ReduceOp.BAND`，
- `dist.ReduceOp.BOR`，
- `dist.ReduceOp.BXOR`，
- `dist.ReduceOp.PREMUL_SUM`。

支持的操作符的完整列表可以在 [这里](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)找到。

除了`dist.all_reduce(tensor, op, group)`之外，PyTorch中还实现了许多其他集合。以下是一些支持的集合。

- `dist.broadcast(tensor, src, group)`：将`tensor`从`src`复制到所有其他进程。
- `dist.reduce(tensor, dst, op, group)`：将`op`应用于每个`tensor`，并将结果存储在`dst`中。
- `dist.all_reduce(tensor, op, group)`：与reduce相同，但结果存储在所有进程中。
- `dist.scatter(tensor, scatter_list, src, group)`：将第i个张量`scatter_list[i]`复制到第i个进程。
- `dist.gather(tensor, gather_list, dst, group)`：将所有进程中的`tensor`复制到`dst`。
- `dist.all_gather(tensor_list, tensor, group)`：将所有进程中的`tensor`复制到`tensor_list`，在所有进程中。
- `dist.barrier(group)`：阻塞组中的所有进程，直到每个进程都进入此函数。
- `dist.all_to_all(output_tensor_list, input_tensor_list, group)`：将输入张量列表分散到组中的所有进程，并返回收集的输出张量列表。

支持的集合的完整列表可以通过查看PyTorch分布式的最新文档[(链接)](https://pytorch.org/docs/stable/distributed.html)找到。

## 分布式训练

现在我们已经了解了分布式模块的工作原理，让我们用它来写一些有用的东西。我们的目标将是复制 [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)的功能。当然，这将是一个教学示例，在实际情况中，你应该使用上面链接的官方、经过测试和优化的版本。

非常简单，我们想要实现一个分布式版本的随机梯度下降。我们的脚本将让所有进程计算其模型在其数据批次上的梯度，然后平均它们的梯度。为了确保在更改进程数量时类似的收敛结果，我们首先必须对我们的数据集进行划分。（你也可以使用[torch.utils.data.random_split](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)，而不是下面的代码片段。）

```python
""" 数据集划分助手 """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()  # from random import Random
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
```

通过上面的代码片段，我们现在可以使用以下几行简单地对任何数据集进行划分：

```python
""" 划分MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                             batch_size=bsz,
                                             shuffle=True)
    return train_set, bsz
```

假设我们有2个副本，那么每个进程将有一个`train_set`，包含60000 / 2 = 30000个样本。我们还通过进程数量划分批次大小，以保持*总体*批次大小为128。

我们现在可以编写我们通常的前向-反向-优化训练代码，并添加一个函数调用来平均我们模型的梯度。（以下内容主要受到官方 [PyTorch MNIST示例](https://github.com/pytorch/examples/blob/master/mnist/main.py)的启发。）

```python
""" 分布式同步SGD示例 """
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)
```

剩下要实现的是`average_gradients(model)`函数，它简单地获取一个模型并在整个 world 平均其梯度。

```python
""" 梯度平均。 """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
```

我们成功实现了分布式同步SGD，并可以在大型计算机集群上训练任何模型。

**注意**：虽然最后一句话是*技术上*正确的，但实现生产级的同步SGD还需要[很多技巧](https://seba-1511.github.io/dist_blog)。再次，使用[经过测试和优化的版本](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)。

### 我们自己的Ring-Allreduce

作为一个额外的挑战，想象一下我们想要实现DeepSpeech的高效 ring allreduce。这可以通过使用点对点集合来实现。

```python
""" 使用加法的环形reduce实现。 """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # 发送send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # 发送recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]
```

在上面的脚本中，`allreduce(send, recv)`函数有一个稍微不同的签名。它接受一个`recv`张量，并将所有`send`张量的总和存储在其中。作为一个留给读者的练习，我们的版本与DeepSpeech中的版本还有一个区别：他们的实现将梯度张量分成*块*，以便最佳利用通信带宽。（提示：[torch.chunk](https://pytorch.org/docs/stable/torch.html#torch.chunk)）

## 高级主题

我们现在准备好探索一些更高级的功能`torch.distributed`。由于有很多内容需要涵盖，本节分为两个小节：

1. 通信后端：我们将学习如何使用MPI和Gloo进行GPU-GPU通信。
2. 初始化方法：我们将了解如何最好地设置`dist.init_process_group()`中的初始协调阶段。

### 通信后端

`torch.distributed`最优雅的方面之一是它能够抽象并构建在不同的后端之上。如前所述，PyTorch中实现了多个后端。一些最流行的后端是Gloo，NCCL和 MPI。它们各有不同的规范和权衡，取决于所需的用例。支持的功能的比较表可以在这里找到[(链接)](https://pytorch.org/docs/stable/distributed.html#module-torch.distributed)。

**Gloo后端**

到目前为止，我们已经广泛使用了[Gloo后端](https://github.com/facebookincubator/gloo)。它非常方便作为开发平台，因为它包含在预编译的PyTorch二进制文件中，并且适用于Linux（自0.2版本起）和macOS（自1.3版本起）。它支持CPU上的所有点对点和集合操作，以及GPU上的所有集合操作。GPU张量的集合操作的实现没有NCCL后端提供的优化。

正如你肯定注意到的，如果将`model`放在GPU上，我们的分布式SGD示例将无法工作。为了使用多个GPU，让我们也进行以下修改：

1. 使用`device = torch.device("cuda:{}".format(rank))`
2. model = Net(), → model = Net().to(device)
3. 使用`data, target = data.to(device), target.to(device)`

通过上述修改，我们的模型现在在两个GPU上进行训练，你可以使用`watch nvidia-smi`监控它们的利用率。

**MPI后端**

消息传递接口（MPI）是高性能计算领域中标准化的工具。它允许进行点对点和集合通信，并且是`torch.distributed`API的主要灵感来源。有几个MPI实现（例如[Open-MPI](https://www.open-mpi.org/)，[MVAPICH2](http://mvapich.cse.ohio-state.edu/)，[Intel MPI](https://software.intel.com/en-us/intel-mpi-library)），每个都针对不同的目的进行了优化。使用MPI后端的优势在于MPI的广泛可用性——以及在大型计算机集群上的高度优化。[一些](https://developer.nvidia.com/mvapich) [最近的](https://developer.nvidia.com/ibm-spectrum-mpi) [实现](https://www.open-mpi.org/) 还能够利用CUDA IPC和GPU Direct技术，以避免通过CPU进行内存复制。

不幸的是，PyTorch的二进制文件不能包含MPI实现，我们必须手动重新编译它。幸运的是，这个过程相当简单，因为PyTorch在编译时会*自动*查找可用的MPI实现。以下步骤通过从源代码安装PyTorch来安装MPI后端。

1. 创建并激活你的Anaconda环境，按照 [指南](https://github.com/pytorch/pytorch#from-source)安装所有先决条件，但**不要**运行`python setup.py install`。
2. 选择并安装你喜欢的MPI实现。请注意，启用CUDA-aware MPI可能需要一些额外的步骤。在我们的例子中，我们将坚持使用不带GPU支持的Open-MPI：`conda install -c conda-forge openmpi`
3. 现在，进入你克隆的PyTorch仓库并执行`python setup.py install`。

为了测试我们新安装的后端，需要进行一些修改。

1. 将`if __name__ == '__main__':`下的内容替换为`init_process(0, 0, run, backend='mpi')`。
2. 运行 `mpirun -n 4 python myscript.py`。

这些更改的原因是MPI需要在生成进程之前创建自己的环境。MPI还将生成自己的进程并执行`初始化方法 (#initialization-methods)中描述的握手`，使`rank`和`size`参数对于`init_process_group`来说是多余的。这实际上非常强大，因为你可以将额外的参数传递给`mpirun`，以便为每个进程定制计算资源。（例如，每个进程的核心数量，手动分配机器到特定秩，以及[更多](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile)）这样做，你应该得到与其他通信后端相同的熟悉输出。

**NCCL后端**

[NCCL后端](https://github.com/nvidia/nccl) 提供了针对CUDA张量的集合操作的优化实现。如果你只对集合操作使用CUDA张量，请考虑使用此后端以获得最佳的性能。NCCL后端包含在支持CUDA的预构建二进制文件中。

### 初始化方法

为了结束本教程，让我们检查一下我们调用的初始化函数：`dist.init_process_group(backend, init_method)`。特别是，我们将讨论各种初始化方法，这些方法负责在每个进程之间的初步协调步骤。这些方法使你能够定义如何完成这种协调。

初始化方法的选择取决于你的硬件设置，一种方法可能比其他方法更合适。除了以下部分，请参考[官方文档](https://pytorch.org/docs/stable/distributed.html#initialization)以获取更多信息。

**环境变量**

我们一直在使用环境变量初始化方法。通过在所有机器上设置以下四个环境变量，所有进程将能够正确连接到主进程，获取有关其他进程的信息。

- `MASTER_PORT`：主进程所在机器上的一个空闲端口。
- `MASTER_ADDR`：主进程所在机器的IP地址。
- `WORLD_SIZE`： 总进程数，以便主进程知道要等待多少个工作进程。
- `RANK`：每个进程的秩，以便它们知道它是否是主进程或工作进程。

### 共享文件系统

共享文件系统要求所有进程能够访问共享文件系统，并将通过共享文件协调它们。这意味着每个进程将打开文件，写入其信息，并等待直到每个人都这样做。之后，所有需要的信息将立即可用于所有进程。为了防止竞争条件，文件系统必须支持通过`fcntl`进行锁定。

```python
dist.init_process_group(
    init_method='file:///mnt/nfs/sharedfile',
    rank=args.rank,
    world_size=4)
```

### TCP

通过TCP进行初始化可以通过提供秩为0的进程的IP地址和一个可达的端口号来实现。在这里，所有工作进程都能够连接到秩为0的进程，并交换如何相互连接的信息。

```python
dist.init_process_group(
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4)
```
