param(
    [Parameter(Mandatory = $true)]
    [string]$ModelStore,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeDirectory
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
. (Join-Path $PSScriptRoot "runtime-qualification-attestation.common.ps1")

$repoRoot = Get-RepositoryRoot
$schemaPath = Join-Path $repoRoot "schemas/runtime-qualification-attestation.schema.json"
$manifestPath = Join-Path $repoRoot "crates/impossible-ocr-onnx/runtime/onnxruntime-1.28.0.json"
$modelStorePath = [IO.Path]::GetFullPath($ModelStore)
$runtimePath = [IO.Path]::GetFullPath($RuntimeDirectory)
if (-not [IO.Path]::IsPathFullyQualified($modelStorePath) -or -not [IO.Path]::IsPathFullyQualified($runtimePath)) {
    throw "model and runtime directories must be absolute"
}

Push-Location $repoRoot
try {
    $exitCode = 1
    $lines = @(& cargo run --locked --offline -p impossible-ocr-server --features onnx-runtime `
        --bin qualify-runtime -- `
        --model-store $modelStorePath `
        --runtime-manifest $manifestPath `
        --runtime-directory $runtimePath `
        --platform windows-x86_64)
    $exitCode = $LASTEXITCODE
}
finally { Pop-Location }

if ($exitCode -ne 0 -or $lines.Count -ne 1) {
    throw "real Windows runtime qualification did not emit one successful report"
}
try { $report = $lines[0] | ConvertFrom-Json -Depth 64 -ErrorAction Stop }
catch { throw "real Windows runtime qualification emitted invalid JSON" }

$attestation = New-RuntimeQualificationAttestation $repoRoot $report
$json = $attestation | ConvertTo-Json -Depth 64
if (-not (Test-Json -Json $json -SchemaFile $schemaPath -ErrorAction SilentlyContinue)) {
    throw "generated runtime qualification attestation failed its closed schema"
}
Assert-RuntimeQualificationAttestation $repoRoot ($json | ConvertFrom-Json -Depth 64)

$outputDirectory = Join-Path $repoRoot "attestations/runtime"
[IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$outputPath = Join-Path $outputDirectory $script:AttestationName
$temporaryPath = Join-Path $outputDirectory ".$($script:AttestationName).$PID.tmp"
try {
    [IO.File]::WriteAllText($temporaryPath, "$json`n", [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporaryPath -Destination $outputPath -Force
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) { Remove-Item -LiteralPath $temporaryPath -Force }
}

Write-Output "Recorded privacy-safe Windows runtime qualification attestation."
