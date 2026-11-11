param(
    [switch]$DiagOnly,
    [switch]$PreviewOnly,
    [switch]$AudioRemuxOnly,
    [string]$EarshotDir = $env:EARSHOT_DIR,
    [string]$EarshotPy  = $env:EARSHOT_PYTHON
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

if (-not $EarshotPy -and $EarshotDir) {
    $CandidatePy = Join-Path $EarshotDir ".venv\Scripts\python.exe"
    if (Test-Path $CandidatePy) { $EarshotPy = $CandidatePy }
}
$PythonCmd = if ($EarshotPy -and (Test-Path $EarshotPy)) { $EarshotPy } else { "python" }

if (-not $AudioRemuxOnly) {
    Write-Host "=== [1/4] Building CUDA + C++20 Simulation (-O3 --use_fast_math) ==="
    cmd.exe /c "$ProjectDir\build.bat"
    if ($LASTEXITCODE -ne 0) { throw "Build failed!" }

    $Exe = "$ProjectDir\build\lotus_rain_sim.exe"

    Write-Host "=== [2/4] Running Fast GPU Diagnostic Test (--diag) ==="
    & $Exe --diag
    if ($LASTEXITCODE -ne 0) { throw "GPU diagnostic test failed!" }
    if ($DiagOnly) { exit 0 }

    if ($PreviewOnly) {
        Write-Host "=== [3/4] Running 6-Keyframe Full-Res Preview + Coupled 20s WAV (--preview) ==="
        & $Exe --preview --out-dir $ProjectDir
        if ($LASTEXITCODE -ne 0) { throw "Preview run failed!" }
        & $PythonCmd "$ProjectDir\verify_qa.py" $ProjectDir
        exit $LASTEXITCODE
    }

    Write-Host "=== [3/4] Rendering Full 1920x1080 @ 60fps (1,200 Frames) + Coupled 48kHz 24-bit WAV ==="
    & $Exe --out-dir $ProjectDir
    if ($LASTEXITCODE -ne 0) { throw "Full cinema render failed!" }
}

if ($EarshotPy -and (Test-Path $EarshotPy)) {
    Write-Host "=== [Optional] Synthesizing Coupled Earshot Procedural Rain Soundtrack & Remuxing MP4 ==="
    if ($EarshotDir -and (Test-Path $EarshotDir)) { Push-Location $EarshotDir }
    & $EarshotPy "$ProjectDir\synthesize_earshot_audio.py" --events "$ProjectDir\gpu_impact_events.bin" --out "$ProjectDir\lotus_rain_65536.wav"
    $SynthCode = $LASTEXITCODE
    if ($EarshotDir -and (Test-Path $EarshotDir)) { Pop-Location }
    if ($SynthCode -ne 0) { throw "Earshot audio synthesis failed!" }

    $RemuxTmp = "$ProjectDir\lotus_rain_65536_remux.mp4"
    ffmpeg -y -loglevel error -i "$ProjectDir\lotus_rain_65536_cuda.mp4" -i "$ProjectDir\lotus_rain_65536.wav" `
        -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -ar 48000 -shortest $RemuxTmp
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg remux failed!" }
    Move-Item -Force $RemuxTmp "$ProjectDir\lotus_rain_65536_cuda.mp4"
}

Write-Host "=== [4/4] Running Mandatory Visual & Audio Verification Gate ==="
& $PythonCmd "$ProjectDir\verify_qa.py" $ProjectDir
if ($LASTEXITCODE -ne 0) { throw "Verification gate failed!" }

Write-Host "=== ALL GATES PASSED: $ProjectDir\lotus_rain_65536_cuda.mp4 ==="
