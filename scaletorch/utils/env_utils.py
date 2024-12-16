import logging
import platform
import socket

import psutil
import torch
from transformers.utils import (is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_npu_available)

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_system_info():
    logger.info('系统诊断信息:')
    info = {
        '操作系统': platform.platform(),
        'Python 版本': platform.python_version(),
        'CPU 核心数': psutil.cpu_count(),
        '可用内存': psutil.virtual_memory().available / (1024**3),
        '网络接口': {
            interface:
            [addr.address for addr in addrs if addr.family == socket.AF_INET]
            for interface, addrs in psutil.net_if_addrs().items()
        },
    }
    # CUDA 和 PyTorch 信息
    dl_info = {
        'CUDA 可用': torch.cuda.is_available(),
        'CUDA 版本': torch.version.cuda,
        'CUDA 设备数': torch.cuda.device_count(),
        'cuDNN 版本': torch.backends.cudnn.version(),
        'PyTorch 版本': torch.__version__,
    }
    info.update(dl_info)

    if is_torch_bf16_gpu_available():
        info['PyTorch version'] += ' (BF16)'
        info['BF16 GPU available'] = 'True'

    if is_torch_cuda_available():
        info['PyTorch version'] += ' (GPU)'
        info['GPU type'] = torch.cuda.get_device_name()

    if is_torch_npu_available():
        info['PyTorch version'] += ' (NPU)'
        info['NPU type'] = torch.npu.get_device_name()
        info['CANN version'] = torch.version.cann

    try:
        import deepspeed  # type: ignore

        info['DeepSpeed version'] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes

        info['Bitsandbytes version'] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm

        info['vLLM version'] = vllm.__version__
    except Exception:
        pass

    for key, value in info.items():
        logger.info(f'{key}: {value}')

    return info


if __name__ == '__main__':
    get_system_info()
