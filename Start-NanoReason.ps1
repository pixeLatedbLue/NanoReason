param(
    [ValidateSet('SFT', 'GRPO', 'Base', 'Demo')]
    [string]$Model = 'GRPO',
    [string]$Adapter,
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,
    [string]$Python = (Join-Path $env:LOCALAPPDATA 'NanoReason\inference\.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Python runtime not found. Pass -Python with a GPU environment containing NanoReason dependencies.'
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the existing server or choose -Port."
}
if ($Adapter -and $Model -in @('Base', 'Demo')) { throw '-Adapter is only valid for SFT or GRPO.' }
if (-not $Adapter) {
    if ($Model -eq 'SFT') { $Adapter = 'artifacts\research\pilot\sft-initial' }
    if ($Model -eq 'GRPO') { $Adapter = 'artifacts\slm\original-grpo' }
}
if ($Adapter) {
    if (-not [IO.Path]::IsPathRooted($Adapter)) { $Adapter = Join-Path $taskRoot $Adapter }
    foreach ($taskFile in @('adapter_config.json', 'adapter_model.safetensors')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Adapter $taskFile))) {
            throw "Missing $taskFile in $Adapter. Restore that checkpoint first."
        }
    }
}
$taskEnvironment = @{}
foreach ($taskName in @('PYTHONPATH', 'NANOREASON_ENGINE', 'NANOREASON_CONFIG', 'NANOREASON_ADAPTER',
                        'NANOREASON_RESULTS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')) {
    $taskEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}
Push-Location -LiteralPath $taskRoot
try {
    $env:PYTHONPATH = Join-Path $taskRoot 'src'
    $env:NANOREASON_ENGINE = if ($Model -eq 'Demo') { 'demo' } else { 'transformers' }
    $env:NANOREASON_CONFIG = Join-Path $taskRoot 'configs\local_inference.toml'
    $env:NANOREASON_ADAPTER = $Adapter
    $env:NANOREASON_RESULTS = Join-Path $taskRoot 'artifacts\slm\evaluation\runs'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    Write-Host "NanoReason $Model : http://127.0.0.1:$Port"
    Write-Host 'Loading before serving. Do not run GPU evaluation concurrently. Ctrl+C stops the server.'
    & $Python -m nanoreason.serve.app --port $Port --preload
    if ($LASTEXITCODE -ne 0) { throw "NanoReason exited with code $LASTEXITCODE" }
} finally {
    foreach ($taskName in $taskEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskName, $taskEnvironment[$taskName], 'Process')
    }
    Pop-Location
}
