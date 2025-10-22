# START_OLLAMA_CPU_MODE.ps1
# Starts Ollama in CPU-only mode (no GPU)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Ollama in CPU-Only Mode" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Stop any running Ollama processes
Write-Host "[1/3] Stopping existing Ollama processes..." -ForegroundColor Yellow
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "[OK] Ollama stopped`n" -ForegroundColor Green

# Step 2: Set environment variables for CPU-only mode
Write-Host "[2/3] Setting CPU-only environment variables..." -ForegroundColor Yellow
$env:OLLAMA_NUM_GPU = "0"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OLLAMA_HOST = "127.0.0.1:11434"
Write-Host "[OK] Environment configured:`n" -ForegroundColor Green
Write-Host "  OLLAMA_NUM_GPU = 0 (No GPU)" -ForegroundColor Gray
Write-Host "  CUDA_VISIBLE_DEVICES = -1 (No CUDA)`n" -ForegroundColor Gray

# Step 3: Start Ollama service in CPU mode
Write-Host "[3/3] Starting Ollama service..." -ForegroundColor Yellow
$ollamaPath = "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe"

if (-Not (Test-Path $ollamaPath)) {
    Write-Host "[ERROR] Ollama not found at: $ollamaPath" -ForegroundColor Red
    exit 1
}

Start-Process -FilePath $ollamaPath -ArgumentList "serve" -NoNewWindow
Start-Sleep -Seconds 5

# Verify Ollama is running
Write-Host "`n[OK] Verifying Ollama is running..." -ForegroundColor Yellow
try {
    $result = & $ollamaPath list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Ollama is running successfully!`n" -ForegroundColor Green
        Write-Host "Available models:" -ForegroundColor Cyan
        & $ollamaPath list
    } else {
        Write-Host "[WARNING] Ollama may not be fully started yet" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Could not verify Ollama status" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Ollama is running in CPU-ONLY mode" -ForegroundColor Green
Write-Host "  URL: http://127.0.0.1:11434" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Press any key to keep this window open..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
