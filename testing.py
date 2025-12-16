import torch

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current device name: {torch.cuda.get_device_name(0)}")