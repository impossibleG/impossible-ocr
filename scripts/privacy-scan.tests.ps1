$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scannerSource = Join-Path $repoRoot "scripts/privacy-scan.ps1"
$headerSource = Join-Path $repoRoot "docs/assets/impossible-ocr-header.png"
$script:Passed = 0

function Assert-Scan([scriptblock]$Mutate, [bool]$ShouldPass, [string]$Description) {
    $fixture = Join-Path ([IO.Path]::GetTempPath()) ("impossible-ocr-privacy-" + [Guid]::NewGuid().ToString("N"))
    try {
        [IO.Directory]::CreateDirectory((Join-Path $fixture "scripts")) | Out-Null
        [IO.Directory]::CreateDirectory((Join-Path $fixture "docs/assets")) | Out-Null
        Copy-Item -LiteralPath $scannerSource -Destination (Join-Path $fixture "scripts/privacy-scan.ps1")
        Copy-Item -LiteralPath $headerSource -Destination (Join-Path $fixture "docs/assets/impossible-ocr-header.png")
        & $Mutate $fixture
        & git -C $fixture init --quiet 2>$null
        if ($LASTEXITCODE -ne 0) { throw "unable to initialize privacy fixture" }
        $output = @(& pwsh -NoProfile -File (Join-Path $fixture "scripts/privacy-scan.ps1") 2>&1)
        $passed = $LASTEXITCODE -eq 0
        if ($passed -ne $ShouldPass) { throw "$Description produced an unexpected privacy result: $($output -join ' ')" }
        $script:Passed++
    }
    finally {
        if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
    }
}

Assert-Scan {} $true "reviewed header"
Assert-Scan {
    param($Root)
    $path = Join-Path $Root "docs/assets/impossible-ocr-header.png"
    $bytes = [IO.File]::ReadAllBytes($path)
    $bytes[$bytes.Length - 1] = $bytes[$bytes.Length - 1] -bxor 1
    [IO.File]::WriteAllBytes($path, $bytes)
} $false "wrong header digest"
Assert-Scan {
    param($Root)
    Move-Item -LiteralPath (Join-Path $Root "docs/assets/impossible-ocr-header.png") -Destination (Join-Path $Root "docs/assets/renamed.png")
} $false "wrong header path"
Assert-Scan {
    param($Root)
    [IO.File]::WriteAllBytes((Join-Path $Root "docs/assets/additional.png"), [byte[]](0, 1, 2, 3))
} $false "additional binary"

Write-Output "PASS: $script:Passed privacy binary allowlist assertions"
