import platform
import psutil
import socket
import torch


def system_diagnostic():
    print("系统诊断信息:")
    print(f"操作系统: {platform.platform()}")
    print(f"Python 版本: {platform.python_version()}")
    print(f"CPU 核心数: {psutil.cpu_count()}")
    print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # 网络接口信息
    print("\n网络接口:")
    for interface, addrs in psutil.net_if_addrs().items():
        print(f"{interface}:")
        for addr in addrs:
            if addr.family == socket.AF_INET:
                print(f"  IP地址: {addr.address}")

    # CUDA 和 PyTorch 信息
    print("\nCUDA 和 PyTorch 信息:")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 设备数: {torch.cuda.device_count()}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")