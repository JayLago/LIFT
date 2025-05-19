"""
    @author: Jay Lago, NIWC Pacific, 55280
"""
import torch

def get_quantile_index(quantile_list, q):
    return min(range(len(quantile_list)), key=lambda i: abs(quantile_list[i] - q))

def toNumpyCPU(arr):
    return arr.cpu().detach().numpy()

def toCPU(arr):
    return arr.cpu().detach()

def get_cuda_summary(memory=False):
    device = 'cpu'
    if torch.cuda.is_available():
        print('CUDA device = ', torch.cuda.get_device_name())
        print('Available number of devices = ', torch.cuda.device_count())
        print('Device numbers              = ', list(range(torch.cuda.device_count())))
        print('Current device              = ', torch.cuda.current_device())
        if memory:
            print(torch.cuda.memory_summary())
        device = f'cuda:{torch.cuda.current_device()}'
    
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("[WARNING] MPS available but not built")
        else:
            device = 'mps'
    else:
        print('[WARNING] GPU is not available!')
    
    return device
