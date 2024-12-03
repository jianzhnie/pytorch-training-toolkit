# MNIST Example

## Single GPU
```bash
python singlegpu.py
```


## Multi-GPUs
```bash
python multigpu.py
```


## Torchrun 
```bash
torchrun  --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d examples/mnist/multigpu_torchrun.py
```

