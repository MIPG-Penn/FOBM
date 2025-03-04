
import torch

def test_gpu_available():
    # print(f'PyTorch version: {torch.__version__}')
    # print(f'cuda.get_device_name: {torch.cuda.get_device_name(0)}')
    # print(f'cuda.current_device: {torch.cuda.current_device()}')
    # print(f'cuda_getCompiledVersion: {torch._C._cuda_getCompiledVersion()}')
    assert(torch.cuda.is_available())