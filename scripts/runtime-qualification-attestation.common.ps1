$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$script:AttestationName = "windows-x86_64-onnxruntime-1.28.0.json"
$script:AttestationKind = "impossible-ocr-runtime-qualification-attestation"
$script:Profile = "paddlex-ocr-3.7-max960"
$script:Hex64 = '^[0-9a-f]{64}$'

function Get-RepositoryRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ExactPropertyNames([object]$Value) {
    if ($Value -is [Collections.IDictionary]) { return @($Value.Keys) }
    return @($Value.PSObject.Properties.Name)
}

function Assert-ExactProperties([object]$Value, [string[]]$Expected, [string]$Context) {
    $actual = @(Get-ExactPropertyNames $Value)
    if ($actual.Count -ne $Expected.Count) { throw "$Context has an unexpected property count" }
    for ($index = 0; $index -lt $Expected.Count; $index++) {
        if ($actual[$index] -cne $Expected[$index]) { throw "$Context has unexpected or reordered properties" }
    }
}

function Get-SourceFiles([string]$RepositoryRoot) {
    $paths = [Collections.Generic.List[string]]::new()
    foreach ($relative in @("Cargo.lock", "Cargo.toml", "rust-toolchain.toml")) {
        $path = Join-Path $RepositoryRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "qualification source input is missing" }
        $paths.Add($relative)
    }
    foreach ($subtree in @("crates", "vendor/impossible-server/crates")) {
        $root = Join-Path $RepositoryRoot $subtree
        foreach ($file in Get-ChildItem -LiteralPath $root -Recurse -File) {
            $relative = [IO.Path]::GetRelativePath($RepositoryRoot, $file.FullName).Replace('\', '/')
            if ($relative.StartsWith("crates/impossible-ocr-onnx/runtime/", [StringComparison]::Ordinal)) {
                continue
            }
            if ($file.Name -ceq "Cargo.toml" -or $file.Extension -cin @(".rs", ".proto", ".json", ".yaml")) {
                $paths.Add($relative)
            }
        }
    }
    $ordered = $paths.ToArray()
    [Array]::Sort($ordered, [StringComparer]::Ordinal)
    return @($ordered | ForEach-Object { Join-Path $RepositoryRoot $_ })
}

function Get-QualificationSourceDigest([string]$RepositoryRoot) {
    $hash = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        foreach ($path in Get-SourceFiles $RepositoryRoot) {
            $relative = [IO.Path]::GetRelativePath($RepositoryRoot, $path).Replace('\', '/')
            $name = [Text.Encoding]::UTF8.GetBytes("$relative`n")
            $text = [IO.File]::ReadAllText($path, [Text.UTF8Encoding]::new($false, $true))
            $bytes = [Text.Encoding]::UTF8.GetBytes($text.Replace("`r`n", "`n").Replace("`r", "`n"))
            $length = [BitConverter]::GetBytes([UInt64]$bytes.LongLength)
            if ([BitConverter]::IsLittleEndian) { [Array]::Reverse($length) }
            $hash.AppendData($name)
            $hash.AppendData($length)
            $hash.AppendData($bytes)
        }
        return [Convert]::ToHexString($hash.GetHashAndReset()).ToLowerInvariant()
    }
    finally { $hash.Dispose() }
}

function Get-WindowsRuntimeBinding([string]$RepositoryRoot) {
    $manifestPath = Join-Path $RepositoryRoot "crates/impossible-ocr-onnx/runtime/onnxruntime-1.28.0.json"
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -Depth 64
    $platforms = @($manifest.platforms | Where-Object { $_.id -ceq "windows-x86_64" })
    if ($platforms.Count -ne 1) { throw "the Windows runtime platform record is missing or duplicated" }
    $platform = $platforms[0]
    $files = @($platform.requiredFiles)
    if ($files.Count -ne 5) { throw "the Windows runtime platform must bind exactly five promoted files" }
    foreach ($file in $files) {
        if ($null -eq $file.bytes -or [string]$file.sha256 -cnotmatch $script:Hex64) {
            throw "the Windows runtime platform is provisional"
        }
    }
    $binding = [ordered]@{
        name = [string]$manifest.runtime.name
        version = [string]$manifest.runtime.version
        source_commit = [string]$manifest.runtime.sourceCommit
        c_api_version = [int]$manifest.runtime.cApiVersion
        execution_provider = [string]$manifest.runtime.executionProvider
        platform_id = [string]$platform.id
        target_triple = [string]$platform.targetTriple
        archive_bytes = [UInt64]$platform.archive.bytes
        archive_sha256 = [string]$platform.archive.sha256
        required_files = @($files | ForEach-Object {
            [ordered]@{
                install_name = [string]$_.installName
                bytes = [UInt64]$_.bytes
                sha256 = [string]$_.sha256
            }
        })
    }
    $canonical = $binding | ConvertTo-Json -Depth 16 -Compress
    $canonicalBytes = [Text.Encoding]::UTF8.GetBytes($canonical)
    $platformDigest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($canonicalBytes)).ToLowerInvariant()
    return [ordered]@{
        platform_record_sha256 = $platformDigest
        identity = $binding
    }
}

function Assert-CuratedCatalogLiterals([string]$RepositoryRoot) {
    $source = Get-Content -LiteralPath (Join-Path $RepositoryRoot "crates/impossible-ocr-onnx/src/lifecycle.rs") -Raw
    foreach ($pattern in @(
        '"detector-inference\.yml",\s*903,\s*"98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699"',
        '"english-recognizer-inference\.yml",\s*3_964,\s*"27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"'
    )) {
        if ([regex]::Matches($source, $pattern).Count -ne 1) { throw "the curated model catalog no longer matches the attestation contract" }
    }
}

function Get-ModelBinding([string]$RepositoryRoot) {
    Assert-CuratedCatalogLiterals $RepositoryRoot
    $detectorPath = Join-Path $RepositoryRoot "crates/impossible-ocr-onnx/contracts/pp-ocrv5-mobile-detector.json"
    $recognizerPath = Join-Path $RepositoryRoot "crates/impossible-ocr-onnx/contracts/pp-ocrv5-english-mobile-recognizer.json"
    $detector = Get-Content -LiteralPath $detectorPath -Raw | ConvertFrom-Json -Depth 64
    $recognizer = Get-Content -LiteralPath $recognizerPath -Raw | ConvertFrom-Json -Depth 64
    if ($detector.qualification -cne "qualified" -or $recognizer.qualification -cne "qualified") {
        throw "model graph contracts are not qualified"
    }
    return [ordered]@{
        bundle_id = $script:Profile
        artifacts = @(
            [ordered]@{ role = "detector"; kind = "onnx_graph"; filename = "detector.onnx"; bytes = [UInt64]$detector.artifact_byte_length; sha256 = [string]$detector.artifact_sha256 },
            [ordered]@{ role = "detector"; kind = "inference_config"; filename = "detector-inference.yml"; bytes = [UInt64]903; sha256 = "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699" },
            [ordered]@{ role = "english_recognizer"; kind = "onnx_graph"; filename = "english-recognizer.onnx"; bytes = [UInt64]$recognizer.artifact_byte_length; sha256 = [string]$recognizer.artifact_sha256 },
            [ordered]@{ role = "english_recognizer"; kind = "inference_config"; filename = "english-recognizer-inference.yml"; bytes = [UInt64]3964; sha256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067" }
        )
        contracts = @(
            [ordered]@{ role = "detector"; contract_id = [string]$detector.contract_id; file_sha256 = Get-Sha256 $detectorPath },
            [ordered]@{ role = "english_recognizer"; contract_id = [string]$recognizer.contract_id; file_sha256 = Get-Sha256 $recognizerPath }
        )
    }
}

function Assert-SuccessReport([object]$Report) {
    Assert-ExactProperties $Report @("schema_version", "status", "profile", "runtime_version", "warmup", "detector_probes", "recognizer_probes", "end_to_end") "qualification report"
    if ([int]$Report.schema_version -ne 1 -or $Report.status -cne "passed" -or $Report.profile -cne $script:Profile -or $Report.runtime_version -cne "1.28.0" -or $Report.warmup -ne $true) {
        throw "qualification report identity or warm-up result is invalid"
    }
    $expectedDetector = @(@(32, 32), @(320, 960), @(960, 320), @(960, 960))
    $detectors = @($Report.detector_probes)
    if ($detectors.Count -ne $expectedDetector.Count) { throw "detector probe matrix drifted" }
    for ($index = 0; $index -lt $detectors.Count; $index++) {
        $probe = $detectors[$index]
        Assert-ExactProperties $probe @("width", "height", "shape_valid", "probability_invariants", "deterministic") "detector probe"
        if ([int]$probe.width -ne $expectedDetector[$index][0] -or [int]$probe.height -ne $expectedDetector[$index][1] -or $probe.shape_valid -ne $true -or $probe.probability_invariants -ne $true -or $probe.deterministic -ne $true) {
            throw "detector probe result drifted"
        }
    }
    $recognizers = @($Report.recognizer_probes)
    $expectedRecognizer = @(@(1,160), @(1,320), @(1,3200), @(8,160), @(8,320), @(8,3200))
    if ($recognizers.Count -ne $expectedRecognizer.Count) { throw "recognizer probe matrix drifted" }
    for ($index = 0; $index -lt $recognizers.Count; $index++) {
        $probe = $recognizers[$index]
        Assert-ExactProperties $probe @("batch", "requested_width", "admitted_width", "probability_invariants", "deterministic") "recognizer probe"
        $batch = $expectedRecognizer[$index][0]; $width = $expectedRecognizer[$index][1]
        if ([int]$probe.batch -ne $batch -or [int]$probe.requested_width -ne $width) { throw "recognizer probe identity drifted" }
        if ($width -eq 160) {
            if ($null -ne $probe.admitted_width -or $probe.probability_invariants -ne $false -or $probe.deterministic -ne $false) { throw "expected recognizer admission rejection drifted" }
        }
        elseif ([int]$probe.admitted_width -ne $width -or $probe.probability_invariants -ne $true -or $probe.deterministic -ne $true) {
            throw "recognizer probe result drifted"
        }
    }
    Assert-ExactProperties $Report.end_to_end @("generated_png", "valid_result", "deterministic") "end-to-end report"
    if ($Report.end_to_end.generated_png -ne $true -or $Report.end_to_end.valid_result -ne $true -or $Report.end_to_end.deterministic -ne $true) {
        throw "end-to-end qualification result drifted"
    }
}

function New-RuntimeQualificationAttestation([string]$RepositoryRoot, [object]$Report) {
    Assert-SuccessReport $Report
    return [ordered]@{
        schema_version = 1
        kind = $script:AttestationKind
        status = "passed"
        platform_id = "windows-x86_64"
        target_triple = "x86_64-pc-windows-msvc"
        profile = $script:Profile
        ort_crate_version = "2.0.0-rc.13"
        qualification_source_sha256 = Get-QualificationSourceDigest $RepositoryRoot
        runtime = Get-WindowsRuntimeBinding $RepositoryRoot
        model_bundle = Get-ModelBinding $RepositoryRoot
        report = $Report
    }
}

function Assert-NoPrivateStrings([object]$Value, [string]$Context = "attestation") {
    if ($null -eq $Value) { return }
    if ($Value -is [string]) {
        if ($Value -match '(?i)(?:[A-Z]:[\\/]|/(?:home|Users)/|\\\\|file:|https?://|localhost|(?:desktop|laptop|workstation)-[A-Z0-9-]+|S-1-[0-9-]+|machine[_-]?name|user[_-]?name|host[_-]?name)') {
            throw "$Context contains a path, host, URL, or local identity"
        }
        foreach ($localIdentity in @(
            [Environment]::UserName,
            [Environment]::UserDomainName,
            [Environment]::MachineName,
            [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
        )) {
            if (-not [string]::IsNullOrWhiteSpace($localIdentity) -and $localIdentity.Length -ge 3 -and $Value.IndexOf($localIdentity, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw "$Context contains a runtime-derived local identity"
            }
        }
        return
    }
    if ($Value -is [Collections.IDictionary]) {
        foreach ($key in $Value.Keys) { Assert-NoPrivateStrings $Value[$key] "$Context.$key" }
        return
    }
    if ($Value -is [Collections.IEnumerable] -and $Value -isnot [Management.Automation.PSCustomObject]) {
        $index = 0; foreach ($item in $Value) { Assert-NoPrivateStrings $item "$Context[$index]"; $index++ }
        return
    }
    foreach ($property in $Value.PSObject.Properties) { Assert-NoPrivateStrings $property.Value "$Context.$($property.Name)" }
}

function Assert-RuntimeQualificationAttestation([string]$RepositoryRoot, [object]$Attestation) {
    Assert-ExactProperties $Attestation @("schema_version", "kind", "status", "platform_id", "target_triple", "profile", "ort_crate_version", "qualification_source_sha256", "runtime", "model_bundle", "report") "attestation"
    if ([int]$Attestation.schema_version -ne 1 -or $Attestation.kind -cne $script:AttestationKind -or $Attestation.status -cne "passed" -or $Attestation.platform_id -cne "windows-x86_64" -or $Attestation.target_triple -cne "x86_64-pc-windows-msvc" -or $Attestation.profile -cne $script:Profile -or $Attestation.ort_crate_version -cne "2.0.0-rc.13") {
        throw "attestation identity is invalid"
    }
    Assert-NoPrivateStrings $Attestation
    $expectedSource = Get-QualificationSourceDigest $RepositoryRoot
    if ($Attestation.qualification_source_sha256 -cne $expectedSource) { throw "qualification source digest is stale" }
    $expectedRuntime = Get-WindowsRuntimeBinding $RepositoryRoot | ConvertTo-Json -Depth 32 -Compress
    if (($Attestation.runtime | ConvertTo-Json -Depth 32 -Compress) -cne $expectedRuntime) { throw "runtime binding is stale or invalid" }
    $expectedModels = Get-ModelBinding $RepositoryRoot | ConvertTo-Json -Depth 32 -Compress
    if (($Attestation.model_bundle | ConvertTo-Json -Depth 32 -Compress) -cne $expectedModels) { throw "model binding is stale or invalid" }
    Assert-SuccessReport $Attestation.report
}
