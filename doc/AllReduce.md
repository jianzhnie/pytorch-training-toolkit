### AllReduce

#### 什么是分布式深度学习？

目前，深度学习的一个重大挑战是其过程非常耗时。设计深度学习模型需要对大量超参数进行设计空间探索并处理大数据。因此，加速训练过程对于我们的研究和开发至关重要。分布式深度学习是减少训练时间的关键技术之一。
在大型环境中训练分布式深度学习模型时，GPU之间的通信是许多挑战之一。在数据并行同步分布式深度学习中，交换梯度的延迟是一个严重的瓶颈。
分布式深度学习中的通信是如何进行的？为什么通信如此耗时？

#### 分布式深度学习中AllReduce的重要性

在同步数据并行分布式深度学习中，主要的计算步骤是：

1. 在每个GPU上使用小批量数据计算损失函数的梯度。
2. 通过GPU间通信计算梯度的均值。
3. 更新模型。

为了计算均值，我们使用了一种称为“AllReduce”的集合通信操作。
目前，针对GPU集群最快的集合通信库之一是NVIDIA集合通信库：NCCL[3]。它的通信性能远优于MPI，MPI是HPC社区中的事实标准通信库。

#### AllReduce算法

首先，让我们看一下AllReduce算法。AllReduce是一种将所有进程中的目标数组减少到一个数组并将其结果数组返回给所有进程的操作。现在，设$ P $ 为总进程数。每个进程都有一个长度为N的数组，称为 $𝐴𝑝 $。进程𝑝（1≤𝑝≤𝑃）的数组的第𝑖个元素是 $A_{p,i}$。 

结果数组B为：
$B_{i}~~=~~A_{1,i}~~Op~~A_{2,i}~~Op~~…~~Op~~A_{P,i}$

这里，Op 是一个二元运算符。SUM、MAX和MIN经常被使用。在分布式深度学习中，SUM操作用于计算梯度的均值。在本文的其余部分，我们假设归约操作是SUM。图1通过一个 P=4 和 N=4 的例子说明了AllReduce操作的工作原理。

<img src="https://tech.preferred.jp/wp-content/uploads/2018/07/fig_1.png" alt="fig_1" style="zoom:50%;" />



> 图1 AllReduce操作

有几种算法可以实现该操作。例如，一个简单的方法是选择一个进程作为主进程，将所有数组收集到主进程中，在主进程中本地执行归约操作，然后将结果数组分发给其他进程。尽管这种算法简单且易于实现，但它不具备可扩展性。主进程是性能瓶颈，因为其通信和归约成本与总进程数成正比。
更快和更具可扩展性的算法已经被提出。它们通过仔细地在参与者进程之间分配计算和通信来消除瓶颈。
这些算法包括 **Ring-AllReduce** 和 **Rabenseifner** 的算法[4]。
我们将在本博客文章中重点关注Ring-AllReduce算法。该算法也被NCCL[5]和baidu-allreduce[6]采用。

#### Ring-AllReduce

我们假设P是总进程数，每个进程被唯一标识为1到P之间的数字。如图2所示，进程构成一个单环。

<img src="https://tech.preferred.jp/wp-content/uploads/2018/07/fig_2.png" alt="fig_2" style="zoom:50%;" />

>  图2 进程环示例

首先，每个进程将其自己的数组分成P个子数组，我们称之为“块”。设chunk[p]为第p个块。
接下来，我们关注进程[p]。进程将chunk[p]发送到下一个进程，同时从上一个进程接收chunk[p-1]（图3）。

<img src="https://tech.preferred.jp/wp-content/uploads/2018/07/fig_3.png" alt="fig_3" style="zoom:50%;" />

> 图3 每个进程将其chunk[p]发送到下一个进程[p+1]

然后，进程p对收到的chunk[p-1]和其自己的chunk[p-1]执行归约操作，并将归约后的块发送到下一个进程p+1（图4）。

<img src="https://tech.preferred.jp/wp-content/uploads/2018/07/fig_4.png" alt="fig_4" style="zoom:50%;" />

> 图4 每个进程将归约后的块发送到下一个进程

通过重复接收-归约-发送步骤P-1次，每个进程获得结果数组的不同部分（图5）。

<img src="https://tech.preferred.jp/wp-content/uploads/2018/07/fig_5.png" alt="fig_5" style="zoom:50%;" />

> 图5 经过P-1步后，每个进程都有一个归约后的子数组。

换句话说，每个进程将其本地块添加到接收到的块并将其发送到下一个进程。换句话说，每个块绕环一周，并在每个进程中累积一个块。访问所有进程一次后，它成为最终结果数组的一部分，最后一个访问的进程持有该块。
最后，所有进程可以通过共享分布式部分结果来获得完整的数组。这是通过再次进行循环步骤而不进行归约操作来实现的，即仅将接收到的块覆盖到每个进程的相应本地块中。当所有进程获得最终数组的所有部分时，AllReduce操作完成。

让我们比较一下Ring-AllReduce与前面提到的简单算法的通信量。

- 在简单算法中，主进程从所有其他进程接收所有数组，这意味着接收的总数据量为(𝑃–1)×𝑁。在归约操作之后，它将数组发送回所有进程，这又是(𝑃–1)×𝑁数据。因此，主进程的通信量与P成正比。
- 在Ring-AllReduce算法中，我们可以按以下方式计算每个进程的通信量。在前半部分算法中，每个进程发送一个大小为𝑁/𝑃的数组，发送𝑃−1次。接下来，每个进程再次发送相同大小的数组𝑃−1次。整个算法中每个进程发送的总数据量为2𝑁(𝑃−1)/𝑃，这实际上与P无关。
  因此，Ring-Allreduce算法比简单算法更高效，因为它通过在所有参与者进程之间均匀分布计算和通信来消除瓶颈进程。许多AllReduce实现采用了Ring-AllReduce，它也适用于分布式深度学习工作负载。



#### 参考文献

[1] [Preferred Networks officially released ChainerMN version 1.0.0](https://www.preferred-networks.jp/en/news/pr20170901)

[2] Akiba, et al., “Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes”

[3] [NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl)

[4] Rabenseifner, “Optimization of Collective Reduction Operations”, ICCS 2004

[5] Jeaugey, [“Optimized Inter-GPU Collective Operations with NCCL”](http://on-demand.gputechconf.com/gtc/2017/presentation/s7155-jeaugey-nccl.pdf), GTC 2017

[6] [baidu-allreduce](https://github.com/baidu-research/baidu-allreduce)

[7] [Open MPI](https://www.open-mpi.org/)

[8] [New ChainerMN functions for improved performance in cloud environments and performance testing results on AWS](https://chainer.org/general/2018/05/25/chainermn-v1-3.html)

[9] Tsuzuku, et al., “Variance-based Gradient Compression for Efficient Distributed Deep Learning”, In Proceedings of ICLR 2018 (Workshop Track)