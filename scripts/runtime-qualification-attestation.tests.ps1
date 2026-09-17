param([string]$RepositoryRoot)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
. (Join-Path $PSScriptRoot "runtime-qualification-attestation.common.ps1")

$script:Passed = 0
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:Passed++
}
function Assert-Throws([scriptblock]$Operation, [string]$Pattern) {
    try { & $Operation; throw "operation unexpectedly succeeded" }
    catch {
        if ($_.Exception.Message -eq "operation unexpectedly succeeded" -or $_.Exception.Message -notmatch $Pattern) { throw }
    }
    $script:Passed++
}
function Copy-Document([object]$Value) {
    return ($Value | ConvertTo-Json -Depth 64) | ConvertFrom-Json -Depth 64
}

$repoRoot = if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) { Get-RepositoryRoot } else { (Resolve-Path $RepositoryRoot).Path }
$schemaPath = (Resolve-Path (Join-Path $PSScriptRoot "../schemas/runtime-qualification-attestation.schema.json")).Path
$realAttestationPath = Join-Path $repoRoot "attestations/runtime/$script:AttestationName"
$realAttestationBefore = if (Test-Path -LiteralPath $realAttestationPath -PathType Leaf) { Get-Sha256 $realAttestationPath } else { $null }
$report = [ordered]@{
    schema_version = 1
    status = "passed"
    profile = "paddlex-ocr-3.7-max960"
    runtime_version = "1.28.0"
    warmup = $true
    detector_probes = @(
        [ordered]@{ width = 32; height = 32; shape_valid = $true; probability_invariants = $true; deterministic = $true },
        [ordered]@{ width = 320; height = 960; shape_valid = $true; probability_invariants = $true; deterministic = $true },
        [ordered]@{ width = 960; height = 320; shape_valid = $true; probability_invariants = $true; deterministic = $true },
        [ordered]@{ width = 960; height = 960; shape_valid = $true; probability_invariants = $true; deterministic = $true }
    )
    recognizer_probes = @(
        [ordered]@{ batch = 1; requested_width = 160; admitted_width = $null; probability_invariants = $false; deterministic = $false },
        [ordered]@{ batch = 1; requested_width = 320; admitted_width = 320; probability_invariants = $true; deterministic = $true },
        [ordered]@{ batch = 1; requested_width = 3200; admitted_width = 3200; probability_invariants = $true; deterministic = $true },
        [ordered]@{ batch = 8; requested_width = 160; admitted_width = $null; probability_invariants = $false; deterministic = $false },
        [ordered]@{ batch = 8; requested_width = 320; admitted_width = 320; probability_invariants = $true; deterministic = $true },
        [ordered]@{ batch = 8; requested_width = 3200; admitted_width = 3200; probability_invariants = $true; deterministic = $true }
    )
    end_to_end = [ordered]@{ generated_png = $true; valid_result = $true; deterministic = $true }
}

$attestation = New-RuntimeQualificationAttestation $repoRoot $report
$json = $attestation | ConvertTo-Json -Depth 64
Assert-True (Test-Json -Json $json -SchemaFile $schemaPath -ErrorAction SilentlyContinue) "valid synthetic attestation failed schema"
Assert-RuntimeQualificationAttestation $repoRoot ($json | ConvertFrom-Json -Depth 64)
Assert-True $true "valid synthetic attestation failed semantic verification"

$unknown = Copy-Document $attestation
$unknown | Add-Member -NotePropertyName unexpected -NotePropertyValue $true
$unknownJson = $unknown | ConvertTo-Json -Depth 64
Assert-True (-not (Test-Json -Json $unknownJson -SchemaFile $schemaPath -ErrorAction SilentlyContinue)) "schema accepted an unknown field"

$stale = Copy-Document $attestation
$stale.qualification_source_sha256 = "0" * 64
Assert-Throws { Assert-RuntimeQualificationAttestation $repoRoot $stale } "source digest is stale"

$runtimeDrift = Copy-Document $attestation
$runtimeDrift.runtime.identity.required_files[0].bytes++
Assert-Throws { Assert-RuntimeQualificationAttestation $repoRoot $runtimeDrift } "runtime binding"

$modelDrift = Copy-Document $attestation
$modelDrift.model_bundle.contracts[0].file_sha256 = "0" * 64
Assert-Throws { Assert-RuntimeQualificationAttestation $repoRoot $modelDrift } "model binding"

$probeDrift = Copy-Document $attestation
$probeDrift.report.detector_probes[0].width = 33
Assert-Throws { Assert-RuntimeQualificationAttestation $repoRoot $probeDrift } "detector probe result drifted"

$rejectionDrift = Copy-Document $attestation
$rejectionDrift.report.recognizer_probes[0].admitted_width = 160
Assert-Throws { Assert-RuntimeQualificationAttestation $repoRoot $rejectionDrift } "expected recognizer admission rejection drifted"

$private = Copy-Document $attestation
$private.runtime.identity.name = "C:\\Users\\private\\onnxruntime.dll"
Assert-Throws { Assert-RuntimeQualificationAttestation $repoRoot $private } "path, host, URL, or local identity"

if ($null -eq $realAttestationBefore) {
    Assert-True (-not (Test-Path -LiteralPath $realAttestationPath)) "native-free tests must not create a passed attestation"
}
else {
    Assert-True ((Get-Sha256 $realAttestationPath) -ceq $realAttestationBefore) "native-free tests must not modify the real attestation"
}
Write-Output "PASS: $script:Passed native-free runtime attestation assertions"
