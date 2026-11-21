import torch
import sys

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    
    # Force create a model on GPU
    import torchvision
    model = torchvision.models.video.r2plus1d_18(pretrained=True).cuda()
    print(f"Model on: {next(model.parameters()).device}")
    
    # Test data transfer
    data = torch.randn(2, 3, 32, 112, 112).cuda()
    print(f"Data on: {data.device}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(data)
        print(f"Output on: {output.device}")
        print("GPU test successful!")
else:
    print("ERROR: CUDA not available")