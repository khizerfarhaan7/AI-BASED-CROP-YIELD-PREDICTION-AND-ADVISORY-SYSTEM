import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

assert torch.cuda.is_available()

print(f"Torch version: {torch.__version__}")
print(f"CUDA availability: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"CUDA device count: {torch.cuda.device_count()}")

tensor = torch.randn(1000, 1000).cuda()

print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

