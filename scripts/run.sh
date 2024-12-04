torchrun  --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d torchddp/elastic_ddp.py
torchrun  --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d mingpt/main.py


torchrun  --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d examples/mnist/multigpu_mnist.py
torchrun  --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d examples/mnist/torchrun_mnist.py
