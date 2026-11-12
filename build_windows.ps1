<#
.SYNOPSIS
    Build the Windows version of Swirl Explorer: dist\SwirlExplorer\SwirlExplorer.exe and a zip.

.DESCRIPTION
    1. Makes a clean virtual environment in .venv-build (unless -UseCurrentPython) and installs
       requirements.txt plus PyInstaller.
    2. Runs the unit tests (unless -SkipTests).
    3. Builds the one-folder app with packaging\SwirlExplorer.spec.
    4. Runs the frozen self-test (unless -SkipSelfTest): a window opens for about 15 s and the
       solver, a Σ-map on worker processes, the 3D view, an MP4 export and a session round trip
       are checked; the report is written to build\selftest\selftest.txt.
    5. Zips the folder to dist\SwirlExplorer-<version>-win64.zip.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
.EXAMPLE
    .\build_windows.ps1 -UseCurrentPython -SkipTests
#>
param(
    [switch]$UseCurrentPython,   # build with the python on PATH instead of a fresh venv
    [switch]$SkipTests,
    [switch]$SkipSelfTest,
    [string]$Python = "python"   # interpreter used to create the venv (3.11 or newer)
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. environment
if ($UseCurrentPython) {
    $py = (Get-Command $Python).Source
    Step "Using $py"
    & $py -m pip install --quiet --disable-pip-version-check pyinstaller
} else {
    $venv = Join-Path $root ".venv-build"
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        Step "Creating the build environment in .venv-build"
        & $Python -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment" }
    }
    $py = Join-Path $venv "Scripts\python.exe"
    Step "Installing requirements"
    & $py -m pip install --quiet --disable-pip-version-check -r requirements.txt pyinstaller pytest
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}
$version = (& $py -c "import swirl_explorer; print(swirl_explorer.__version__)").Trim()
Step "Swirl Explorer $version"

# ---------------------------------------------------------------- 2. tests
if (-not $SkipTests) {
    Step "Running the tests"
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "tests failed; fix them or pass -SkipTests" }
}

# ---------------------------------------------------------------- 3. build
if (-not (Test-Path "packaging\swirl.ico")) { & $py packaging\make_icon.py }
Step "Building with PyInstaller (a few minutes)"
& $py -m PyInstaller --noconfirm --clean --log-level WARN --distpath dist --workpath build\pyinstaller `
    packaging\SwirlExplorer.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
$exe = Join-Path $root "dist\SwirlExplorer\SwirlExplorer.exe"
if (-not (Test-Path $exe)) { throw "no exe at $exe" }

# ---------------------------------------------------------------- 4. frozen self-test
if (-not $SkipSelfTest) {
    Step "Self-test of the frozen app (a window appears briefly)"
    $st = Join-Path $root "build\selftest"
    if (Test-Path $st) { Remove-Item -Recurse -Force $st }
    $p = Start-Process -FilePath $exe -ArgumentList @("--selftest", "`"$st`"") -PassThru -Wait
    Get-Content -Encoding UTF8 (Join-Path $st "selftest.txt") | Write-Host
    if ($p.ExitCode -ne 0) { throw "the frozen self-test failed (see build\selftest\selftest.txt)" }
}

# ---------------------------------------------------------------- 5. zip
$zip = Join-Path $root "dist\SwirlExplorer-$version-win64.zip"
Step "Zipping to $zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $root "dist\SwirlExplorer") -DestinationPath $zip -CompressionLevel Optimal
$mb = [math]::Round((Get-ChildItem -Recurse (Join-Path $root "dist\SwirlExplorer") | Measure-Object Length -Sum).Sum / 1MB)
$zmb = [math]::Round((Get-Item $zip).Length / 1MB)
Step "Done: dist\SwirlExplorer ($mb MB), zip $zmb MB"
Write-Host "Run it:  dist\SwirlExplorer\SwirlExplorer.exe"
