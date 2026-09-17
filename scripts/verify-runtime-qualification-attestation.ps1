$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
. (Join-Path $PSScriptRoot "runtime-qualification-attestation.common.ps1")

$repoRoot = Get-RepositoryRoot
$schemaPath = Join-Path $repoRoot "schemas/runtime-qualification-attestation.schema.json"
$attestationDirectory = Join-Path $repoRoot "attestations/runtime"
$attestationPath = Join-Path $attestationDirectory $script:AttestationName

$attestations = @(if (Test-Path -LiteralPath $attestationDirectory -PathType Container) {
    Get-ChildItem -LiteralPath $attestationDirectory -File -Filter "*.json"
})
if ($attestations.Count -ne 1 -or -not (Test-Path -LiteralPath $attestationPath -PathType Leaf)) {
    throw "exactly one fresh Windows runtime qualification attestation is required"
}

$json = Get-Content -LiteralPath $attestationPath -Raw
if (-not (Test-Json -Json $json -SchemaFile $schemaPath -ErrorAction SilentlyContinue)) {
    throw "runtime qualification attestation failed its closed schema"
}
$attestation = $json | ConvertFrom-Json -Depth 64
Assert-RuntimeQualificationAttestation $repoRoot $attestation
Write-Output "Verified one privacy-safe Windows runtime qualification attestation."
