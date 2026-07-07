Write-Host "Waiting for Level 6 Sniper Ablation (Llama-3) to finish..."
# Wait for all existing python processes to close to free up VRAM
$pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($pythonProcs) {
    $pythonProcs | Wait-Process
}
Write-Host "VRAM cleared. Starting overnight Multi-Model sweep..."
python run_multi_model.py
