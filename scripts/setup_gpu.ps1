param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Installing CUDA-enabled PyTorch stack for RTX GPUs..." -ForegroundColor Cyan
& $Python -m pip install --upgrade pip
& $Python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing project package and YOLO dependencies..." -ForegroundColor Cyan
& $Python -m pip install -e .

Write-Host "GPU setup complete. Verify with:" -ForegroundColor Green
Write-Host "$Python -c ""import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"""
