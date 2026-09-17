$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Assert-ExactBansPolicy([string]$DenyConfig) {
    $bansBlocks = [regex]::Matches(
        $DenyConfig,
        '(?ms)^\[bans\]\s*\r?\n(?<body>.*?)(?=^\[[^\]]+\]|\z)'
    )
    if ($bansBlocks.Count -ne 1) { throw "deny.toml must contain exactly one bans policy" }
    $bansBody = $bansBlocks[0].Groups['body'].Value
    foreach ($requiredPolicy in @(
        '(?m)^\s*multiple-versions\s*=\s*"deny"\s*(?:#.*)?$',
        '(?m)^\s*wildcards\s*=\s*"deny"\s*(?:#.*)?$',
        '(?m)^\s*deny\s*=\s*\[\s*\]\s*(?:#.*)?$',
        '(?m)^\s*skip-tree\s*=\s*\[\s*\]\s*(?:#.*)?$'
    )) {
        if ([regex]::Matches($bansBody, $requiredPolicy).Count -ne 1) {
            throw "cargo-deny bans policy was weakened or duplicated"
        }
    }
    $skipBlocks = [regex]::Matches(
        $bansBody,
        '(?ms)^\s*skip\s*=\s*\[(?<entries>.*?)^\s*\]\s*(?:#.*)?$'
    )
    if ($skipBlocks.Count -ne 1) { throw "cargo-deny must contain one explicit skip list" }
    $skipText = $skipBlocks[0].Groups['entries'].Value
    $entryPattern = '\{\s*name\s*=\s*"(?<name>[A-Za-z0-9_-]+)"\s*,\s*version\s*=\s*"(?<version>=[^"]+)"\s*\}'
    $skipEntries = [regex]::Matches($skipText, $entryPattern)
    $actualSkips = @($skipEntries | ForEach-Object {
        "$($_.Groups['name'].Value)@$($_.Groups['version'].Value)"
    } | Sort-Object)
    $expectedSkips = @("bitflags@=1.3.2", "getrandom@=0.2.17", "syn@=3.0.5")
    if ($skipEntries.Count -ne 3 -or
        (Compare-Object -ReferenceObject $expectedSkips -DifferenceObject $actualSkips)) {
        throw "cargo-deny must contain only the three exact reviewed duplicate exceptions"
    }
    $unparsedSkipText = [regex]::Replace($skipText, $entryPattern, '')
    $unparsedSkipText = [regex]::Replace($unparsedSkipText, '(?m)#.*$', '')
    $unparsedSkipText = [regex]::Replace($unparsedSkipText, '[\s,]', '')
    if ($unparsedSkipText.Length -ne 0) { throw "cargo-deny skip list contains unreviewed syntax" }
}

function New-PolicyMutant([string]$Source, [string]$Before, [string]$After) {
    if ([regex]::Matches($Source, [regex]::Escape($Before)).Count -ne 1) {
        throw "dependency policy mutant source is not unique"
    }
    return $Source.Replace($Before, $After, [StringComparison]::Ordinal)
}

function Assert-PolicyMutantRejected([string]$Mutant) {
    $accepted = $false
    try { Assert-ExactBansPolicy $Mutant; $accepted = $true } catch { }
    if ($accepted) { throw "dependency policy verifier accepted a weakened mutant" }
}

Push-Location $repoRoot
try {
    $metadataJson = cargo metadata --locked --all-features --format-version 1
    if ($LASTEXITCODE -ne 0) { throw "cargo metadata failed" }
    $metadata = $metadataJson | ConvertFrom-Json
    $packages = @($metadata.packages)

    function Assert-ExactPackage([string]$Name, [string]$Version) {
        $matches = @($packages | Where-Object { $_.name -ceq $Name })
        if ($matches.Count -ne 1 -or [string]$matches[0].version -cne $Version) {
            throw "dependency policy requires exactly ${Name} ${Version}"
        }
        return $matches[0]
    }

    $null = Assert-ExactPackage "url" "2.5.8"
    $idna = Assert-ExactPackage "idna" "1.1.0"
    $null = Assert-ExactPackage "idna_adapter" "1.2.0"
    $image = Assert-ExactPackage "image" "0.25.6"
    $null = Assert-ExactPackage "indexmap" "2.11.4"
    $null = Assert-ExactPackage "hashbrown" "0.15.5"
    $null = Assert-ExactPackage "flate2" "1.1.9"
    $null = Assert-ExactPackage "miniz_oxide" "0.8.9"
    $null = Assert-ExactPackage "ort" "2.0.0-rc.13"
    $null = Assert-ExactPackage "ort-sys" "2.0.0-rc.13"
    if ([version]$idna.version -lt [version]"1.0.3") {
        throw "idna must remain at or above the safe 1.0.3 floor"
    }

    $nodeById = @{}
    foreach ($node in $metadata.resolve.nodes) { $nodeById[[string]$node.id] = $node }
    $imageFeatures = @($nodeById[[string]$image.id].features | Sort-Object)
    if (Compare-Object -ReferenceObject @("jpeg", "png") -DifferenceObject $imageFeatures) {
        throw "image must enable exactly the jpeg and png features"
    }

    $icuPackages = @($packages | Where-Object { $_.name -clike "icu_*" })
    if ($icuPackages.Count -eq 0) { throw "the reviewed ICU4X graph is missing" }
    foreach ($package in $icuPackages) {
        if (-not ([string]$package.version).StartsWith("1.", [StringComparison]::Ordinal)) {
            throw "ICU4X major-version drift detected: $($package.name) $($package.version)"
        }
        if ($null -eq $package.rust_version -or [version]$package.rust_version -gt [version]"1.88.0") {
            throw "ICU4X MSRV drift detected: $($package.name) requires $($package.rust_version)"
        }
    }

    $tooNew = @($packages | Where-Object {
        $null -ne $_.rust_version -and [version]$_.rust_version -gt [version]"1.88.0"
    })
    if ($tooNew.Count -ne 0) {
        $summary = ($tooNew | ForEach-Object { "$($_.name) $($_.version) requires $($_.rust_version)" }) -join "; "
        throw "dependency MSRV exceeds Rust 1.88: $summary"
    }

    $onnxPackage = Assert-ExactPackage "impossible-ocr-onnx" "0.1.0"
    $ortDependency = @($onnxPackage.dependencies | Where-Object { $_.name -ceq "ort" })
    if ($ortDependency.Count -ne 1 -or -not $ortDependency[0].optional -or
        $ortDependency[0].uses_default_features -or
        (Compare-Object -ReferenceObject @("api-28", "load-dynamic", "std") -DifferenceObject @($ortDependency[0].features | Sort-Object))) {
        throw "impossible-ocr-onnx must keep ort optional with exactly api-28, load-dynamic, and std"
    }
    $defaultTree = cargo tree --locked -p impossible-ocr-onnx --no-default-features --prefix none
    if ($LASTEXITCODE -ne 0) { throw "default impossible-ocr-onnx dependency tree failed" }
    if (@($defaultTree | Where-Object { $_ -match '^ort(?:-sys)? v' }).Count -ne 0) {
        throw "the default impossible-ocr-onnx graph must not activate ONNX Runtime"
    }

    $prohibited = @($packages | Where-Object {
        $_.name -match '^(?:opencv|ffmpeg|ffmpeg-sys|geos|geos-sys|clipper|clipper2|geo)$'
    })
    if ($prohibited.Count -ne 0) {
        throw "native, geometry, or ONNX runtime dependency entered the Wave 1 graph"
    }

    $denyConfig = Get-Content -LiteralPath (Join-Path $repoRoot "deny.toml") -Raw
    Assert-ExactBansPolicy $denyConfig
    $policyMutants = @(
        (New-PolicyMutant $denyConfig 'version = "=1.3.2"' 'version = "=1.3.1"'),
        (New-PolicyMutant $denyConfig 'name = "syn"' 'name = "smallvec"'),
        (New-PolicyMutant $denyConfig 'multiple-versions = "deny"' 'multiple-versions = "allow"'),
        (New-PolicyMutant $denyConfig 'skip-tree = []' 'skip-tree = [{ name = "bitflags", version = "=2.13.2" }]')
    )
    foreach ($mutant in $policyMutants) { Assert-PolicyMutantRejected $mutant }

    $expectedDuplicates = [ordered]@{
        bitflags = @("1.3.2", "2.13.2")
        getrandom = @("0.2.17", "0.3.4")
        syn = @("2.0.119", "3.0.5")
    }
    $duplicateRows = foreach ($target in @("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc")) {
        $tree = cargo tree --locked --all-features --duplicates --depth 0 --target $target
        if ($LASTEXITCODE -ne 0) { throw "cargo tree failed for $target" }
        foreach ($line in $tree) {
            if ($line -match '^([A-Za-z0-9_-]+) v([^ ]+)$') {
                [pscustomobject]@{ name = $Matches[1]; version = $Matches[2] }
            }
        }
    }
    $actualDuplicates = @{}
    foreach ($group in ($duplicateRows | Group-Object name)) {
        $versions = @($group.Group.version | Sort-Object -Unique)
        if ($versions.Count -gt 1) { $actualDuplicates[$group.Name] = $versions }
    }
    if ($actualDuplicates.ContainsKey("smallvec")) {
        throw "the optional ONNX Runtime graph introduces an unreviewed smallvec duplicate family"
    }
    if (Compare-Object -ReferenceObject @($expectedDuplicates.Keys) -DifferenceObject @($actualDuplicates.Keys)) {
        throw "reviewed duplicate dependency families changed"
    }
    foreach ($name in $expectedDuplicates.Keys) {
        if (Compare-Object -ReferenceObject $expectedDuplicates[$name] -DifferenceObject $actualDuplicates[$name]) {
            throw "reviewed duplicate versions changed for $name"
        }
    }

    Write-Output "Dependency policy verified: safe IDNA, ICU4X 1.x, Rust 1.88, codecs, optional dynamic ORT rc.13, unified compression/map graphs, prohibited stacks, and three exact reviewed duplicate exceptions."
} finally {
    Pop-Location
}
