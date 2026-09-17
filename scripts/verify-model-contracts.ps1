$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$schemaPath = Join-Path $repoRoot "schemas/onnx-model-contract.schema.json"
$contractsPath = Join-Path $repoRoot "crates/impossible-ocr-onnx/contracts"

if (-not (Test-Json -LiteralPath $schemaPath)) {
    throw "ONNX model contract schema is not valid JSON"
}

$contracts = @(Get-ChildItem -LiteralPath $contractsPath -Filter "*.json" -File | Sort-Object Name)
if ($contracts.Count -ne 2) {
    throw "exactly two curated ONNX model contracts are required"
}

foreach ($contract in $contracts) {
    if (-not (Test-Json -LiteralPath $contract.FullName -SchemaFile $schemaPath)) {
        throw "ONNX model contract failed schema validation"
    }
}

function Assert-SchemaRejects([object]$Document, [string]$Description) {
    $json = $Document | ConvertTo-Json -Depth 32 -Compress
    if (Test-Json -Json $json -SchemaFile $schemaPath -ErrorAction SilentlyContinue) {
        throw "ONNX model contract schema accepted mutant: $Description"
    }
}

$qualifiedWithoutGraph = Get-Content -LiteralPath $contracts[0].FullName -Raw | ConvertFrom-Json
$qualifiedWithoutGraph.ir_version = $null
Assert-SchemaRejects $qualifiedWithoutGraph "qualified contract without measured graph metadata"

$unknownField = Get-Content -LiteralPath $contracts[0].FullName -Raw | ConvertFrom-Json
$unknownField | Add-Member -NotePropertyName "unexpected" -NotePropertyValue $true
Assert-SchemaRejects $unknownField "unknown field"

$invalidDigest = Get-Content -LiteralPath $contracts[0].FullName -Raw | ConvertFrom-Json
$invalidDigest.artifact_sha256 = $invalidDigest.artifact_sha256.ToUpperInvariant()
Assert-SchemaRejects $invalidDigest "non-canonical digest"

$invalidLength = Get-Content -LiteralPath $contracts[0].FullName -Raw | ConvertFrom-Json
$invalidLength.artifact_byte_length = 0
Assert-SchemaRejects $invalidLength "non-positive artifact byte length"

Write-Output "Validated two curated ONNX contracts and four schema mutants."
