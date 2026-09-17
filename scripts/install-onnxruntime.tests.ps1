$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$manifestPath = Join-Path $PSScriptRoot "..\crates\impossible-ocr-onnx\runtime\onnxruntime-1.28.0.json"
$manifestSchema = Join-Path $PSScriptRoot "..\schemas\onnxruntime-runtime-manifest.schema.json"
if (-not (Test-Json -LiteralPath $manifestSchema)) { throw "native runtime manifest schema is invalid" }
if (-not (Test-Json -LiteralPath $manifestPath -SchemaFile $manifestSchema)) {
    throw "native runtime manifest failed schema validation"
}

. (Join-Path $PSScriptRoot "install-onnxruntime.ps1")

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

function New-TestManifest([string]$Id) {
    $source = Join-Path $PSScriptRoot "..\crates\impossible-ocr-onnx\runtime\onnxruntime-1.28.0.json"
    $manifest = Get-Content -LiteralPath $source -Raw | ConvertFrom-Json -Depth 32
    $selected = @($manifest.platforms | Where-Object { $_.id -ceq $Id })[0]
    $manifest.platforms = @($selected)
    return $manifest
}

function Get-TestContent([string]$Member) {
    return [Text.Encoding]::UTF8.GetBytes("reviewed synthetic content for $Member`n")
}

function Set-QualifiedFiles($PlatformRecord) {
    foreach ($file in $PlatformRecord.requiredFiles) {
        $bytes = Get-TestContent ([string]$file.member)
        $hash = [Security.Cryptography.SHA256]::HashData($bytes)
        $file.bytes = $bytes.Length
        $file.sha256 = [Convert]::ToHexString($hash).ToLowerInvariant()
    }
}

function New-TestZip($PlatformRecord, [string]$Path, [hashtable]$ExtraEntries) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        $null = $zip.CreateEntry(([string]$PlatformRecord.archive.rootDirectory + '/'))
        foreach ($file in $PlatformRecord.requiredFiles) {
            $name = [string]$PlatformRecord.archive.rootDirectory + '/' + [string]$file.member
            $entry = $zip.CreateEntry($name, [IO.Compression.CompressionLevel]::NoCompression)
            $output = $entry.Open()
            try { $bytes = Get-TestContent ([string]$file.member); $output.Write($bytes, 0, $bytes.Length) }
            finally { $output.Dispose() }
        }
        foreach ($pair in $ExtraEntries.GetEnumerator()) {
            $entry = $zip.CreateEntry([string]$pair.Key, [IO.Compression.CompressionLevel]::NoCompression)
            $output = $entry.Open()
            try { $bytes = [Text.Encoding]::UTF8.GetBytes([string]$pair.Value); $output.Write($bytes, 0, $bytes.Length) }
            finally { $output.Dispose() }
        }
    }
    finally { $zip.Dispose(); $stream.Dispose() }
}

function New-DuplicateZip($PlatformRecord, [string]$Path) {
    New-TestZip $PlatformRecord $Path @{}
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Update, $false)
    try {
        $name = [string]$PlatformRecord.archive.rootDirectory + '/license'
        $entry = $zip.CreateEntry($name)
        $output = $entry.Open()
        try { $output.WriteByte(1) } finally { $output.Dispose() }
    }
    finally { $zip.Dispose(); $stream.Dispose() }
}

function New-SymlinkZip($PlatformRecord, [string]$Path) {
    New-TestZip $PlatformRecord $Path @{}
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Update, $false)
    try {
        $entry = $zip.CreateEntry(([string]$PlatformRecord.archive.rootDirectory + '/lib/link.dll'))
        $entry.ExternalAttributes = (0xA1FF -shl 16)
        $output = $entry.Open()
        try { $bytes = [Text.Encoding]::UTF8.GetBytes('onnxruntime.dll'); $output.Write($bytes, 0, $bytes.Length) }
        finally { $output.Dispose() }
    }
    finally { $zip.Dispose(); $stream.Dispose() }
}

function New-TestTarGz($PlatformRecord, [string]$Path, [switch]$WithReviewedLinks, [switch]$WithLink) {
    $file = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $gzip = [IO.Compression.GZipStream]::new($file, [IO.Compression.CompressionLevel]::NoCompression, $false)
    $writer = [System.Formats.Tar.TarWriter]::new($gzip, [System.Formats.Tar.TarEntryFormat]::Ustar, $false)
    try {
        foreach ($required in $PlatformRecord.requiredFiles) {
            $name = [string]$PlatformRecord.archive.rootDirectory + '/' + [string]$required.member
            $entry = [System.Formats.Tar.UstarTarEntry]::new([System.Formats.Tar.TarEntryType]::RegularFile, $name)
            $entry.DataStream = [IO.MemoryStream]::new((Get-TestContent ([string]$required.member)), $false)
            try { $writer.WriteEntry($entry) } finally { $entry.DataStream.Dispose() }
        }
        if ($WithReviewedLinks) {
            foreach ($reviewed in @($PlatformRecord.allowedMemberRules.ignoredLinks)) {
                $link = [System.Formats.Tar.UstarTarEntry]::new([System.Formats.Tar.TarEntryType]::SymbolicLink,
                    ([string]$PlatformRecord.archive.rootDirectory + '/' + [string]$reviewed.member))
                $link.LinkName = [string]$reviewed.target
                $writer.WriteEntry($link)
            }
        }
        if ($WithLink) {
            $link = [System.Formats.Tar.UstarTarEntry]::new([System.Formats.Tar.TarEntryType]::SymbolicLink,
                ([string]$PlatformRecord.archive.rootDirectory + '/lib/libonnxruntime.so'))
            $link.LinkName = 'libonnxruntime.so.1.28.0'
            $writer.WriteEntry($link)
        }
    }
    finally { $writer.Dispose(); $gzip.Dispose(); $file.Dispose() }
}

function Set-ArchiveIdentity($PlatformRecord, [string]$Path) {
    $item = Get-Item -LiteralPath $Path
    $PlatformRecord.archive.bytes = $item.Length
    $PlatformRecord.archive.sha256 = Get-Sha256 $Path
}

$temp = Join-Path ([IO.Path]::GetTempPath()) ('impossible-ocr-ort-tests-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($temp) | Out-Null
try {
    $presentLength = [Nullable[long]]::new(17)
    Assert-True (Test-ResponseLength $presentLength 17) 'boxed response length was rejected'
    Assert-True (Test-ResponseLength $null 17) 'absent response length was rejected'
    Assert-True (-not (Test-ResponseLength ([long]18) 17)) 'incorrect response length was accepted'

    $production = New-TestManifest 'windows-x86_64'
    Assert-True ((Assert-Manifest $production 'windows-x86_64' $true).id -ceq 'windows-x86_64') `
        'qualified production Windows manifest was rejected'
    $record = Assert-Manifest $production 'windows-x86_64' $false
    Assert-True ($record.archive.assetId -eq 489173573) 'official Windows asset identity changed'

    $manifest = New-TestManifest 'windows-x86_64'
    $platformRecord = $manifest.platforms[0]
    Set-QualifiedFiles $platformRecord
    $validZip = Join-Path $temp 'valid.zip'
    New-TestZip $platformRecord $validZip @{}
    Set-ArchiveIdentity $platformRecord $validZip
    $stage = Join-Path $temp 'zip-stage'; [IO.Directory]::CreateDirectory($stage) | Out-Null
    $derived = @(Expand-SafeZip $platformRecord $validZip $stage)
    Assert-DerivedFiles $platformRecord $derived
    Assert-True ($derived.Count -eq $platformRecord.requiredFiles.Count) 'valid ZIP did not produce every required file'
    Assert-True ((Get-ChildItem -LiteralPath $stage -File).Count -eq $derived.Count) 'ZIP extracted unexpected files'

    $traversal = Join-Path $temp 'traversal.zip'
    New-TestZip $platformRecord $traversal @{ ([string]$platformRecord.archive.rootDirectory + '/../escape.dll') = 'bad' }
    $badStage = Join-Path $temp 'bad-stage-1'; [IO.Directory]::CreateDirectory($badStage) | Out-Null
    Assert-Throws { Expand-SafeZip $platformRecord $traversal $badStage } 'traversal|unsafe'

    $unexpected = Join-Path $temp 'unexpected.zip'
    New-TestZip $platformRecord $unexpected @{ ([string]$platformRecord.archive.rootDirectory + '/lib/payload.exe') = 'bad' }
    $badStage = Join-Path $temp 'bad-stage-2'; [IO.Directory]::CreateDirectory($badStage) | Out-Null
    Assert-Throws { Expand-SafeZip $platformRecord $unexpected $badStage } 'unexpected member'

    $duplicate = Join-Path $temp 'duplicate.zip'
    New-DuplicateZip $platformRecord $duplicate
    $badStage = Join-Path $temp 'bad-stage-3'; [IO.Directory]::CreateDirectory($badStage) | Out-Null
    Assert-Throws { Expand-SafeZip $platformRecord $duplicate $badStage } 'case-colliding'

    $symlink = Join-Path $temp 'symlink.zip'
    New-SymlinkZip $platformRecord $symlink
    $badStage = Join-Path $temp 'bad-stage-4'; [IO.Directory]::CreateDirectory($badStage) | Out-Null
    Assert-Throws { Expand-SafeZip $platformRecord $symlink $badStage } 'links and reparse'

    $changed = Join-Path $temp 'changed.zip'
    [IO.File]::WriteAllBytes($changed, [byte[]](1, 2, 3))
    Assert-Throws { Assert-ArchiveFile $platformRecord $changed } 'integrity check failed'

    $linuxManifest = New-TestManifest 'linux-x86_64'
    $linux = $linuxManifest.platforms[0]
    Set-QualifiedFiles $linux
    $validTar = Join-Path $temp 'valid.tgz'
    New-TestTarGz $linux $validTar -WithReviewedLinks
    Set-ArchiveIdentity $linux $validTar
    $tarStage = Join-Path $temp 'tar-stage'; [IO.Directory]::CreateDirectory($tarStage) | Out-Null
    $tarDerived = @(Expand-SafeTarGz $linux $validTar $tarStage)
    Assert-DerivedFiles $linux $tarDerived
    Assert-True ($tarDerived.Count -eq $linux.requiredFiles.Count) 'valid TGZ did not produce every required file'

    $linkTar = Join-Path $temp 'link.tgz'
    New-TestTarGz $linux $linkTar -WithLink
    $badStage = Join-Path $temp 'bad-stage-5'; [IO.Directory]::CreateDirectory($badStage) | Out-Null
    Assert-Throws { Expand-SafeTarGz $linux $linkTar $badStage } 'links and special'

    $manifestPath = Join-Path $temp 'qualified.json'
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 32), [Text.UTF8Encoding]::new($false))
    $Action = 'Install'; $Platform = 'windows-x86_64'; $ManifestPath = $manifestPath
    $ArchivePath = $validZip; $RuntimeRoot = Join-Path $temp 'runtime'; $AllowDownload = $false
    $installed = Invoke-OrtPackaging
    Assert-True ((Test-Path -LiteralPath $installed -PathType Container)) 'qualified runtime was not installed'
    $Action = 'Verify'
    Assert-True ((Invoke-OrtPackaging) -ceq $installed) 'offline installed-runtime verification failed'

    Write-Output "PASS: $script:Passed native runtime packaging assertions"
}
finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force }
}
