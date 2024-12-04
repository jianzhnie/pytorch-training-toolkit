import logging
import os
import platform
import socket
from typing import Optional

import psutil
import torch
import torch.distributed as dist

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def system_diagnostic():
    logger.info('系统诊断信息:')
    logger.info(f'操作系统: {platform.platform()}')
    logger.info(f'Python 版本: {platform.python_version()}')
    logger.info(f'CPU 核心数: {psutil.cpu_count()}')
    logger.info(
        f'可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB')

    # 网络接口信息
    logger.info('\n网络接口:')
    for interface, addrs in psutil.net_if_addrs().items():
        logger.info(f'{interface}:')
        for addr in addrs:
            if addr.family == socket.AF_INET:
                logger.info(f'  IP地址: {addr.address}')

    # CUDA 和 PyTorch 信息
    logger.info('\nCUDA 和 PyTorch 信息:')
    logger.info(f'CUDA 可用: {torch.cuda.is_available()}')
    logger.info(f'CUDA 设备数: {torch.cuda.device_count()}')
    logger.info(f'cuDNN 版本: {torch.backends.cudnn.version()}')
    logger.info(f'CUDA 版本: {torch.version.cuda}')
    logger.info(f'PyTorch 版本: {torch.__version__}')


def validate_distributed_setup() -> bool:
    """Validate distributed environment setup.

    Returns:
        bool: True if distributed environment is correctly configured
    """
    required_env_vars = [
        'MASTER_ADDR',
        'MASTER_PORT',
        'WORLD_SIZE',
        'LOCAL_RANK',
        'RANK',
    ]
    for var in required_env_vars:
        if var not in os.environ:
            logger.error(f'Missing required environment variable: {var}')
            return False
    return True


def ddp_setup(rank: Optional[int] = None,
              world_size: Optional[int] = None) -> None:
    """Set up the Distributed Data Parallel (DDP) environment with
    comprehensive checks.

    Args:
        rank (Optional[int]): Specific rank to set. If None, uses environment variable.
        world_size (Optional[int]): Total number of processes. If None, uses environment variable.

    Raises:
        RuntimeError: If distributed environment is not properly configured
    """
    if not validate_distributed_setup():
        raise RuntimeError(
            'Distributed environment not properly configured. '
            "Ensure you're using torchrun or torch.distributed.launch with correct environment variables."
        )

    try:
        # Use provided values or fall back to environment variables
        current_rank = rank or int(os.environ.get('RANK', 0))
        current_world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))

        # Initialize distributed process group
        dist.init_process_group(backend='nccl',
                                rank=current_rank,
                                world_size=current_world_size)

        # Determine device
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)

        logger.info(
            f'DDP Setup Complete: Rank {current_rank}, World Size {current_world_size}'
        )

    except Exception as e:
        logger.error(f'Failed to setup distributed environment: {e}')
        raise RuntimeError(f'Distributed setup failed: {e}')


def ddp_cleanup() -> None:
    """Safely clean up the distributed environment."""
    try:
        dist.destroy_process_group()
        logger.info('Distributed process group destroyed successfully')
    except RuntimeError as e:
        logger.warning(f'Error during distributed cleanup: {e}')
