import torch

# CUDA 사용 가능 여부 확인
cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# 설치된 CUDA 버전 확인
if cuda_available:
    cuda_version = torch.version.cuda
    print(f"설치된 CUDA 버전: {cuda_version}")
    
    # cuDNN 버전 확인
    cudnn_version = torch.backends.cudnn.version()
    print(f"설치된 cuDNN 버전: {cudnn_version}")
