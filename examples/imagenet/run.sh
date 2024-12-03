# singgpu
python examples/imagenet/dist_train.py -a resnet18 --dummy

# Single node, multiple GPUs:
python examples/imagenet/dist_train.py -a resnet18 --dummy \
    --dist-url 'tcp://127.0.0.1:8080' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --batch-size 32 \
    --world-size 1 \
    --rank 0
