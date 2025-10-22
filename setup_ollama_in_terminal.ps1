# Setup Ollama in Current PowerShell Session
# Run this script to refresh PATH and setup Ollama

Write-Host "🔧 Setting up Ollama in current PowerShell session..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Refresh PATH environment variable
Write-Host "Step 1: Refreshing PATH environment variable..." -ForegroundColor Yellow
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Write-Host "✅ PATH refreshed" -ForegroundColor Green
Write-Host ""

# Step 2: Check if Ollama is installed
Write-Host "Step 2: Checking Ollama installation..." -ForegroundColor Yellow
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (Test-Path $ollamaPath) {
    Write-Host "✅ Ollama is installed at: $ollamaPath" -ForegroundColor Green
} else {
    Write-Host "❌ Ollama is not installed!" -ForegroundColor Red
    Write-Host "📥 Please download and install from: https://ollama.ai/download" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Step 3: Check Ollama version
Write-Host "Step 3: Checking Ollama version..." -ForegroundColor Yellow
try {
    $version = ollama --version
    Write-Host "✅ $version" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot run ollama command" -ForegroundColor Red
    Write-Host "💡 Try closing and reopening PowerShell" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Step 4: Check if mistral model is already installed
Write-Host "Step 4: Checking installed models..." -ForegroundColor Yellow
$modelList = ollama list 2>$null
Write-Host $modelList
Write-Host ""

# Step 5: Prompt to pull Mistral model
Write-Host "Step 5: Mistral Model Setup" -ForegroundColor Yellow
Write-Host "The RAG Coach uses the Mistral 7B model." -ForegroundColor Cyan
Write-Host ""
Write-Host "Available Mistral variants:" -ForegroundColor White
Write-Host "  1. mistral (latest, ~4.1 GB)" -ForegroundColor White
Write-Host "  2. mistral:7b-instruct (recommended, ~4.1 GB)" -ForegroundColor White
Write-Host "  3. mistral:7b-instruct-q4_K_M (4-bit quantized, ~4.4 GB)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Do you want to pull a Mistral model now? (y/n)"
if ($choice -eq 'y' -or $choice -eq 'Y') {
    Write-Host ""
    Write-Host "Which variant do you want?" -ForegroundColor Cyan
    Write-Host "  [1] mistral:latest (default)" -ForegroundColor White
    Write-Host "  [2] mistral:7b-instruct (recommended)" -ForegroundColor White
    Write-Host "  [3] mistral:7b-instruct-q4_K_M (currently downloading in other terminal)" -ForegroundColor White
    Write-Host ""
    
    $modelChoice = Read-Host "Enter choice (1/2/3) [default: 2]"
    
    $modelName = switch ($modelChoice) {
        "1" { "mistral:latest" }
        "3" { "mistral:7b-instruct-q4_K_M" }
        default { "mistral:7b-instruct" }
    }
    
    Write-Host ""
    Write-Host "📥 Pulling model: $modelName" -ForegroundColor Cyan
    Write-Host "⏱️  This will take 5-15 minutes depending on your internet speed..." -ForegroundColor Yellow
    Write-Host ""
    
    ollama pull $modelName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Model downloaded successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Current models:" -ForegroundColor Cyan
        ollama list
    } else {
        Write-Host ""
        Write-Host "❌ Model download failed" -ForegroundColor Red
    }
} else {
    Write-Host "⏭️  Skipping model download" -ForegroundColor Yellow
    Write-Host "💡 You can pull it later with: ollama pull mistral:7b-instruct" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Make sure the Mistral model is downloaded (check with: ollama list)" -ForegroundColor White
Write-Host "  2. Update rag_coach.py to use your chosen model name" -ForegroundColor White
Write-Host "  3. Start your backend: python -m uvicorn backend_api:app --reload" -ForegroundColor White
Write-Host "  4. Start your frontend: streamlit run app.py" -ForegroundColor White
Write-Host ""
