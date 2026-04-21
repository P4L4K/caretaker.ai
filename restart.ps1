# 1. Kill existing processes on ports 8000 and 3000
Write-Host "Stopping existing project processes..." -ForegroundColor Yellow
$connections = Get-NetTCPConnection -LocalPort 8000, 3000 -ErrorAction SilentlyContinue
if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique
    Stop-Process -Id $pids -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped processes: $pids" -ForegroundColor Green
} else {
    Write-Host "No processes found on ports 8000 or 3000." -ForegroundColor Gray
}

# 2. Start Backend
Write-Host "Starting Backend..." -ForegroundColor Cyan
Start-Process cmd -ArgumentList '/k cd /d "C:\Users\ARYAN ANGRAL\Desktop\caretakerai\backend" && "venv\Scripts\activate.bat" && python main.py'

# 3. Start Frontend
Write-Host "Starting Frontend..." -ForegroundColor Cyan
Start-Process cmd -ArgumentList '/k cd /d "C:\Users\ARYAN ANGRAL\Desktop\caretakerai\frontend" && npm run dev'

Write-Host "Project is restarting! Check the new windows for logs." -ForegroundColor Green
