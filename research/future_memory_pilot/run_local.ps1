param(
    [ValidateRange(1, 10000)][int]$Budget = 96,
    [ValidateRange(1, 6)][int]$Limit = 6
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$server = Join-Path $repoRoot 'tools/llama-b10796/bin/llama-server.exe'
$python = Join-Path $repoRoot '.venv-six/Scripts/python.exe'
$modelRoot = Join-Path (Split-Path $repoRoot -Parent) 'pilot_runtime/models'
$chatModel = Join-Path $modelRoot 'qwen35/Qwen_Qwen3.5-0.8B-Q8_0.gguf'
$embedModel = Join-Path $modelRoot 'qwen3-embed/Qwen3-Embedding-0.6B-Q8_0.gguf'
foreach ($requiredPath in @($server, $python, $chatModel, $embedModel)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) { throw "Missing runtime file: $requiredPath" }
}
if (@(Get-NetTCPConnection -State Listen -LocalPort 9911, 9010 -ErrorAction SilentlyContinue).Count) {
    throw 'Port 9911 or 9010 is occupied. Existing servers were not changed.'
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$output = Join-Path $repoRoot "results/future_memory_pilot_$stamp"
New-Item -ItemType Directory -Path $output -ErrorAction Stop | Out-Null
$ownedProcesses = @()
$runExitCode = 1
try {
    $commonArgs = @('--host', '127.0.0.1', '--gpu-layers', 'all', '--parallel', '1', '--threads', '6')
    $chatArgs = @('-m', "`"$chatModel`"", '--alias', 'Qwen3.5-0.8B', '--port', '9911', '--ctx-size', '4096', '--jinja', '--reasoning', 'off') + $commonArgs
    $embedArgs = @('-m', "`"$embedModel`"", '--alias', 'Qwen3-Embedding-0.6B', '--port', '9010', '--ctx-size', '2048', '--embedding', '--pooling', 'last', '--batch-size', '2048', '--ubatch-size', '2048') + $commonArgs
    $ownedProcesses += Start-Process -FilePath $server -ArgumentList $chatArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'chat.stdout.log') -RedirectStandardError (Join-Path $output 'chat.stderr.log')
    $ownedProcesses += Start-Process -FilePath $server -ArgumentList $embedArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'embed.stdout.log') -RedirectStandardError (Join-Path $output 'embed.stderr.log')
    Write-Host "Pilot output: $output"
    $deadline = (Get-Date).AddSeconds(300)
    do {
        if (@($ownedProcesses | Where-Object { $_.HasExited }).Count) { throw "A local server exited. Inspect logs in $output" }
        $ready = $true
        foreach ($port in @(9911, 9010)) {
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2
                if ($health.status -ne 'ok') { $ready = $false }
            } catch { $ready = $false }
        }
        if ($ready) { break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    if (-not $ready) { throw "Servers were not ready within 300 seconds. Inspect logs in $output" }
    & $python (Join-Path $PSScriptRoot 'run_pilot.py') --output (Join-Path $output 'run') --budget $Budget --limit $Limit
    $runExitCode = $LASTEXITCODE
} catch {
    Write-Error $_ -ErrorAction Continue
} finally {
    foreach ($ownedProcess in $ownedProcesses) {
        if (-not $ownedProcess.HasExited) { Stop-Process -Id $ownedProcess.Id -Force -ErrorAction SilentlyContinue }
    }
}
exit $runExitCode
