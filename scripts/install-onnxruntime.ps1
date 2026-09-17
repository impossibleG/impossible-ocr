param(
    [ValidateSet("Inspect", "Install", "Verify")]
    [string]$Action = "Verify",
    [ValidateSet("windows-x86_64", "linux-x86_64")]
    [string]$Platform,
    [string]$ManifestPath = (Join-Path $PSScriptRoot "..\crates\impossible-ocr-onnx\runtime\onnxruntime-1.28.0.json"),
    [string]$ArchivePath,
    [string]$RuntimeRoot,
    [switch]$AllowDownload
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Get-Sha256([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $hash = [System.Security.Cryptography.SHA256]::Create()
        try { return [Convert]::ToHexString($hash.ComputeHash($stream)).ToLowerInvariant() }
        finally { $hash.Dispose() }
    }
    finally { $stream.Dispose() }
}

function Test-LowerSha256([string]$Value) {
    return $Value -cmatch '^[0-9a-f]{64}$'
}

function Assert-SafeRelative([string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name) -or $Name.Contains('\') -or $Name.Contains(':') -or
        $Name.StartsWith('/') -or $Name.Contains("`0")) {
        throw "archive contains an unsafe member name"
    }
    foreach ($segment in $Name.Split('/')) {
        if ($segment -eq '.' -or $segment -eq '..' -or [string]::IsNullOrEmpty($segment)) {
            throw "archive contains traversal or an ambiguous member name"
        }
    }
}

function Get-PlatformRecord($Manifest, [string]$Id) {
    $matches = @($Manifest.platforms | Where-Object { $_.id -ceq $Id })
    if ($matches.Count -ne 1) { throw "manifest must contain exactly one selected platform" }
    return $matches[0]
}

function Get-ArchiveRelativeName($PlatformRecord, [string]$MemberName, [bool]$Directory) {
    $normalized = $MemberName.Replace('\', '/')
    if ($Directory) { $normalized = $normalized.TrimEnd('/') }
    Assert-SafeRelative $normalized
    if ($Directory -and $normalized -ceq [string]$PlatformRecord.archive.rootDirectory) {
        return ''
    }
    $prefix = [string]$PlatformRecord.archive.rootDirectory + '/'
    if (-not $normalized.StartsWith($prefix, [StringComparison]::Ordinal)) {
        throw "archive member is outside the pinned root directory"
    }
    $relative = $normalized.Substring($prefix.Length)
    if ([string]::IsNullOrEmpty($relative)) { throw "archive member is ambiguous" }
    Assert-SafeRelative $relative
    return $relative
}

function Test-AllowedMember($PlatformRecord, [string]$Relative, [bool]$Directory) {
    $rules = $PlatformRecord.allowedMemberRules
    $exactDirectories = @()
    $exactFiles = @()
    if ($null -ne $rules.PSObject.Properties['exactDirectories']) { $exactDirectories = @($rules.exactDirectories) }
    if ($null -ne $rules.PSObject.Properties['exactFiles']) { $exactFiles = @($rules.exactFiles) }
    if ($Directory -and $exactDirectories -ccontains $Relative) { return $true }
    if (-not $Directory -and $exactFiles -ccontains $Relative) { return $true }
    if ($Directory) { return $Relative -ceq 'include' -or $Relative -ceq 'lib' }
    if (@($PlatformRecord.allowedMemberRules.rootFiles) -ccontains $Relative) { return $true }
    if ($Relative.StartsWith('include/', [StringComparison]::Ordinal)) {
        $leaf = $Relative.Substring(8)
        if ($leaf.Contains('/')) { return $false }
        return @($PlatformRecord.allowedMemberRules.includeExtensions) -ccontains [IO.Path]::GetExtension($leaf)
    }
    if ($Relative.StartsWith('lib/', [StringComparison]::Ordinal)) {
        $leaf = $Relative.Substring(4)
        return -not $leaf.Contains('/') -and @($PlatformRecord.allowedMemberRules.libFiles) -ccontains $leaf
    }
    return $false
}

function Test-IgnoredLink($PlatformRecord, [string]$Relative, [string]$Target) {
    $rules = $PlatformRecord.allowedMemberRules
    if ($null -eq $rules.PSObject.Properties['ignoredLinks']) { return $false }
    $matches = @($rules.ignoredLinks | Where-Object {
        ([string]$_.member -ceq $Relative) -and ([string]$_.target -ceq $Target)
    })
    return $matches.Count -eq 1
}

function Assert-Manifest($Manifest, [string]$Id, [bool]$RequireQualified) {
    if ($Manifest.schemaVersion -ne 1 -or $Manifest.runtime.version -cne '1.28.0' -or
        $Manifest.runtime.sourceCommit -cne 'da9b5e364c465de65c49d91e696cd6485270757f' -or
        $Manifest.runtime.cApiVersion -ne 28 -or $Manifest.runtime.executionProvider -cne 'cpu') {
        throw "runtime manifest identity is not the reviewed ONNX Runtime 1.28.0 CPU contract"
    }
    if ($Manifest.release.releaseId -ne 359547054 -or $Manifest.release.immutable -ne $false) {
        throw "release provenance does not match the reviewed mutable GitHub release"
    }
    $platformRecord = Get-PlatformRecord $Manifest $Id
    $archive = $platformRecord.archive
    if ($archive.bytes -le 0 -or -not (Test-LowerSha256 ([string]$archive.sha256)) -or
        $archive.maximumExpandedBytes -le 0 -or $archive.maximumMembers -le 0) {
        throw "archive bounds or digest are invalid"
    }
    $uri = [Uri]$archive.url
    if ($uri.Scheme -cne 'https' -or -not (@($Manifest.networkPolicy.allowedHosts) -ccontains $uri.DnsSafeHost)) {
        throw "archive URL violates the HTTPS host allowlist"
    }
    $requiredNames = @($platformRecord.requiredFiles | ForEach-Object { [string]$_.member })
    foreach ($required in @('LICENSE', 'ThirdPartyNotices.txt', 'Privacy.md')) {
        if (-not ($requiredNames -ccontains $required)) { throw "required redistribution notice is absent" }
    }
    $seenMembers = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $seenInstalls = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $platformRecord.requiredFiles) {
        Assert-SafeRelative ([string]$file.member)
        Assert-SafeRelative ([string]$file.installName)
        if (-not $seenMembers.Add([string]$file.member) -or -not $seenInstalls.Add([string]$file.installName)) {
            throw "required file names collide"
        }
        if (-not (Test-AllowedMember $platformRecord ([string]$file.member) $false)) {
            throw "required file is outside the member allowlist"
        }
        if ($RequireQualified -and ($null -eq $file.bytes -or $file.bytes -le 0 -or
            $null -eq $file.sha256 -or -not (Test-LowerSha256 ([string]$file.sha256)))) {
            throw "runtime manifest is intentionally unqualified; derived file hashes are required"
        }
    }
    $rules = $platformRecord.allowedMemberRules
    foreach ($propertyName in @('exactDirectories', 'exactFiles')) {
        if ($null -ne $rules.PSObject.Properties[$propertyName]) {
            $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
            foreach ($relative in @($rules.$propertyName)) {
                Assert-SafeRelative ([string]$relative)
                if (-not $seen.Add([string]$relative)) { throw "exact archive member rules collide" }
            }
        }
    }
    if ($null -ne $rules.PSObject.Properties['ignoredLinks']) {
        if ([string]$archive.format -cne 'tgz') { throw "ignored links are valid only for reviewed TGZ archives" }
        $seenLinks = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($link in @($rules.ignoredLinks)) {
            Assert-SafeRelative ([string]$link.member)
            Assert-SafeRelative ([string]$link.target)
            if ([string]$link.target -match '/') { throw "ignored link targets must be local archive leaf names" }
            if (-not $seenLinks.Add([string]$link.member) -or $seenMembers.Contains([string]$link.member)) {
                throw "ignored archive link rules collide"
            }
        }
    }
    return $platformRecord
}

function Assert-PublicAddress([Uri]$Uri) {
    if ($Uri.Scheme -cne 'https' -or -not (@($script:Manifest.networkPolicy.allowedHosts) -ccontains $Uri.DnsSafeHost) -or
        -not [string]::IsNullOrEmpty($Uri.UserInfo)) {
        throw "download URI violates the HTTPS host allowlist"
    }
    $addresses = [Net.Dns]::GetHostAddresses($Uri.DnsSafeHost)
    if ($addresses.Count -eq 0) { throw "download host did not resolve" }
    foreach ($address in $addresses) {
        if ([Net.IPAddress]::IsLoopback($address) -or $address.Equals([Net.IPAddress]::Any) -or
            $address.Equals([Net.IPAddress]::IPv6Any) -or $address.IsIPv6LinkLocal -or
            $address.IsIPv6SiteLocal -or $address.IsIPv6Multicast -or $address.IsIPv6UniqueLocal) {
            throw "download host resolved to a non-public address"
        }
        if ($address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork) {
            $b = $address.GetAddressBytes()
            if ($b[0] -eq 0 -or $b[0] -eq 10 -or $b[0] -eq 127 -or $b[0] -ge 224 -or
                ($b[0] -eq 169 -and $b[1] -eq 254) -or ($b[0] -eq 172 -and $b[1] -ge 16 -and $b[1] -le 31) -or
                ($b[0] -eq 192 -and ($b[1] -eq 0 -or $b[1] -eq 2 -or $b[1] -eq 168)) -or
                ($b[0] -eq 198 -and ($b[1] -eq 18 -or $b[1] -eq 19 -or $b[1] -eq 51)) -or
                ($b[0] -eq 203 -and $b[1] -eq 0 -and $b[2] -eq 113) -or
                ($b[0] -eq 100 -and $b[1] -ge 64 -and $b[1] -le 127)) {
                throw "download host resolved to a non-public address"
            }
        }
    }
}

function Test-ResponseLength([object]$ContentLength, [long]$ExpectedLength) {
    return $null -eq $ContentLength -or [long]$ContentLength -eq $ExpectedLength
}

function Receive-PinnedArchive($PlatformRecord, [string]$Destination) {
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false
    $handler.UseCookies = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [Threading.Timeout]::InfiniteTimeSpan
    try {
        $uri = [Uri]$PlatformRecord.archive.url
        for ($redirect = 0; $redirect -le $script:Manifest.networkPolicy.maximumRedirects; $redirect++) {
            Assert-PublicAddress $uri
            $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $uri)
            $request.Headers.UserAgent.ParseAdd('impossible-ocr-runtime-installer/0.1')
            $send = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
            if (-not $send.Wait([TimeSpan]::FromSeconds($script:Manifest.networkPolicy.connectTimeoutSeconds))) {
                $request.Dispose(); throw "runtime archive connection timed out"
            }
            $response = $send.GetAwaiter().GetResult()
            $request.Dispose()
            if ([int]$response.StatusCode -ge 300 -and [int]$response.StatusCode -le 399) {
                $location = $response.Headers.Location
                $response.Dispose()
                if ($null -eq $location) { throw "runtime archive redirect has no location" }
                $uri = if ($location.IsAbsoluteUri) { $location } else { [Uri]::new($uri, $location) }
                continue
            }
            if (-not $response.IsSuccessStatusCode) { $response.Dispose(); throw "runtime archive request failed" }
            if (-not (Test-ResponseLength $response.Content.Headers.ContentLength ([long]$PlatformRecord.archive.bytes))) {
                $response.Dispose(); throw "runtime archive response length differs from the manifest"
            }
            $networkStream = $response.Content.ReadAsStream()
            $output = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            $hasher = [Security.Cryptography.SHA256]::Create()
            try {
                $buffer = [byte[]]::new(65536)
                $total = 0L
                while ($true) {
                    $readTask = $networkStream.ReadAsync($buffer, 0, $buffer.Length)
                    if (-not $readTask.Wait([TimeSpan]::FromSeconds($script:Manifest.networkPolicy.idleReadTimeoutSeconds))) {
                        throw "runtime archive idle-read deadline exceeded"
                    }
                    $count = $readTask.GetAwaiter().GetResult()
                    if ($count -eq 0) { break }
                    if ($total -gt [long]$PlatformRecord.archive.bytes - $count) { throw "runtime archive exceeded its byte limit" }
                    $total += $count
                    if ($total -gt [long]$PlatformRecord.archive.bytes) { throw "runtime archive exceeded its byte limit" }
                    $output.Write($buffer, 0, $count)
                    $null = $hasher.TransformBlock($buffer, 0, $count, $null, 0)
                }
                $null = $hasher.TransformFinalBlock([byte[]]::new(0), 0, 0)
                $output.Flush($true)
                $digest = [Convert]::ToHexString($hasher.Hash).ToLowerInvariant()
                if ($total -ne [long]$PlatformRecord.archive.bytes -or $digest -cne [string]$PlatformRecord.archive.sha256) {
                    throw "runtime archive integrity check failed"
                }
            }
            finally { $hasher.Dispose(); $output.Dispose(); $networkStream.Dispose(); $response.Dispose() }
            return
        }
        throw "runtime archive exceeded the redirect limit"
    }
    finally { $client.Dispose(); $handler.Dispose() }
}

function Assert-ArchiveFile($PlatformRecord, [string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer -and $item.Length -eq [long]$PlatformRecord.archive.bytes -and
        (Get-Sha256 $item.FullName) -ceq [string]$PlatformRecord.archive.sha256) { return $item.FullName }
    throw "runtime archive integrity check failed"
}

function Copy-PinnedArchive($PlatformRecord, [string]$Source, [string]$Destination) {
    $item = Get-Item -LiteralPath $Source -Force
    if ($item.PSIsContainer -or $item.LinkType -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        $item.Length -ne [long]$PlatformRecord.archive.bytes) { throw "runtime archive integrity check failed" }
    $sourceStream = [IO.File]::Open($item.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $destinationStream = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $buffer = [byte[]]::new(65536); $total = 0L
        while (($count = $sourceStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            if ($total -gt [long]$PlatformRecord.archive.bytes - $count) { throw "runtime archive exceeded its byte limit" }
            $total += $count
            $destinationStream.Write($buffer, 0, $count)
            $null = $hasher.TransformBlock($buffer, 0, $count, $null, 0)
        }
        $null = $hasher.TransformFinalBlock([byte[]]::new(0), 0, 0)
        $destinationStream.Flush($true)
        $digest = [Convert]::ToHexString($hasher.Hash).ToLowerInvariant()
        if ($total -ne [long]$PlatformRecord.archive.bytes -or $digest -cne [string]$PlatformRecord.archive.sha256) {
            throw "runtime archive integrity check failed"
        }
        return $Destination
    }
    finally { $hasher.Dispose(); $destinationStream.Dispose(); $sourceStream.Dispose() }
}

function Copy-VerifiedMember([IO.Stream]$SourceStream, [string]$Destination, [long]$ExpectedLength) {
    $output = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $buffer = [byte[]]::new(65536)
        $total = 0L
        while (($count = $SourceStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            if ($total -gt $ExpectedLength - $count) { throw "archive member exceeded its declared length" }
            $total += $count
            if ($total -gt $ExpectedLength) { throw "archive member exceeded its declared length" }
            $output.Write($buffer, 0, $count)
            $null = $hasher.TransformBlock($buffer, 0, $count, $null, 0)
        }
        $null = $hasher.TransformFinalBlock([byte[]]::new(0), 0, 0)
        $output.Flush($true)
        if ($total -ne $ExpectedLength) { throw "archive member length changed during extraction" }
        return [pscustomobject]@{
            bytes = $total
            sha256 = [Convert]::ToHexString($hasher.Hash).ToLowerInvariant()
        }
    }
    finally { $hasher.Dispose(); $output.Dispose() }
}

function Get-RequiredMap($PlatformRecord) {
    $map = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach ($file in $PlatformRecord.requiredFiles) { $map.Add([string]$file.member, $file) }
    return $map
}

function Complete-Extraction($PlatformRecord, $Required, $Found, [string]$Staging) {
    foreach ($file in $PlatformRecord.requiredFiles) {
        if (-not $Found.ContainsKey([string]$file.member)) { throw "archive is missing a required runtime file" }
    }
    return @($PlatformRecord.requiredFiles | ForEach-Object {
        $record = $Found[[string]$_.member]
        [pscustomobject]@{ member = [string]$_.member; installName = [string]$_.installName; bytes = $record.bytes; sha256 = $record.sha256 }
    })
}

function Expand-SafeZip($PlatformRecord, [string]$Archive, [string]$Staging) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $required = Get-RequiredMap $PlatformRecord
    $found = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $expanded = 0L; $members = 0
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($entry in $zip.Entries) {
            $members++
            if ($members -gt [int]$PlatformRecord.archive.maximumMembers) { throw "archive has too many members" }
            $directory = $entry.FullName.EndsWith('/', [StringComparison]::Ordinal)
            $relative = Get-ArchiveRelativeName $PlatformRecord $entry.FullName $directory
            if ($directory -and $relative.Length -eq 0) { continue }
            if (-not $names.Add($relative)) { throw "archive contains duplicate or case-colliding members" }
            $unixType = (($entry.ExternalAttributes -shr 16) -band 0xF000)
            $windowsAttributes = ($entry.ExternalAttributes -band 0xFFFF)
            if ($unixType -eq 0xA000 -or ($windowsAttributes -band 0x400) -ne 0) { throw "archive links and reparse points are forbidden" }
            if (($directory -and $unixType -ne 0 -and $unixType -ne 0x4000) -or
                (-not $directory -and $unixType -ne 0 -and $unixType -ne 0x8000)) {
                throw "archive special members are forbidden"
            }
            if (-not (Test-AllowedMember $PlatformRecord $relative $directory)) { throw "archive contains an unexpected member" }
            if ($directory) { continue }
            if ([long]$entry.Length -gt [long]$PlatformRecord.archive.maximumExpandedBytes - $expanded) { throw "archive exceeds its expanded-byte limit" }
            $expanded += [long]$entry.Length
            if ($expanded -gt [long]$PlatformRecord.archive.maximumExpandedBytes) { throw "archive exceeds its expanded-byte limit" }
            if ($required.ContainsKey($relative)) {
                $entryStream = $entry.Open()
                try { $record = Copy-VerifiedMember $entryStream (Join-Path $Staging ([string]$required[$relative].installName)) ([long]$entry.Length) }
                finally { $entryStream.Dispose() }
                $found.Add($relative, $record)
            }
        }
    }
    finally { $zip.Dispose() }
    return Complete-Extraction $PlatformRecord $required $found $Staging
}

function Expand-SafeTarGz($PlatformRecord, [string]$Archive, [string]$Staging) {
    $required = Get-RequiredMap $PlatformRecord
    $found = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $expanded = 0L; $members = 0
    $file = [IO.File]::OpenRead($Archive)
    $gzip = [IO.Compression.GZipStream]::new($file, [IO.Compression.CompressionMode]::Decompress, $false)
    $reader = [System.Formats.Tar.TarReader]::new($gzip, $false)
    try {
        while ($null -ne ($entry = $reader.GetNextEntry())) {
            $members++
            if ($members -gt [int]$PlatformRecord.archive.maximumMembers) { throw "archive has too many members" }
            $directory = $entry.EntryType -eq [System.Formats.Tar.TarEntryType]::Directory
            $relative = Get-ArchiveRelativeName $PlatformRecord $entry.Name $directory
            if ($directory -and $relative.Length -eq 0) { continue }
            if (-not $names.Add($relative)) { throw "archive contains duplicate or case-colliding members" }
            if ($entry.EntryType -eq [System.Formats.Tar.TarEntryType]::SymbolicLink) {
                if (-not (Test-IgnoredLink $PlatformRecord $relative ([string]$entry.LinkName))) {
                    throw "archive links and special members are forbidden"
                }
                continue
            }
            if (-not $directory -and $entry.EntryType -ne [System.Formats.Tar.TarEntryType]::RegularFile -and
                $entry.EntryType -ne [System.Formats.Tar.TarEntryType]::V7RegularFile) {
                throw "archive links and special members are forbidden"
            }
            if (-not (Test-AllowedMember $PlatformRecord $relative $directory)) { throw "archive contains an unexpected member" }
            if ($directory) { continue }
            if ([long]$entry.Length -gt [long]$PlatformRecord.archive.maximumExpandedBytes - $expanded) { throw "archive exceeds its expanded-byte limit" }
            $expanded += [long]$entry.Length
            if ($expanded -gt [long]$PlatformRecord.archive.maximumExpandedBytes) { throw "archive exceeds its expanded-byte limit" }
            if ($required.ContainsKey($relative)) {
                if ($null -eq $entry.DataStream) { throw "required archive member has no data" }
                $record = Copy-VerifiedMember $entry.DataStream (Join-Path $Staging ([string]$required[$relative].installName)) ([long]$entry.Length)
                $found.Add($relative, $record)
            }
        }
    }
    finally { $reader.Dispose(); $gzip.Dispose(); $file.Dispose() }
    return Complete-Extraction $PlatformRecord $required $found $Staging
}

function Expand-SafeArchive($PlatformRecord, [string]$Archive, [string]$Staging) {
    if ($PlatformRecord.archive.format -ceq 'zip') { return Expand-SafeZip $PlatformRecord $Archive $Staging }
    if ($PlatformRecord.archive.format -ceq 'tgz') { return Expand-SafeTarGz $PlatformRecord $Archive $Staging }
    throw "unsupported runtime archive format"
}

function Assert-DerivedFiles($PlatformRecord, $Derived) {
    foreach ($actual in $Derived) {
        $expected = @($PlatformRecord.requiredFiles | Where-Object { $_.member -ceq $actual.member })[0]
        if ($null -eq $expected -or $actual.bytes -ne [long]$expected.bytes -or $actual.sha256 -cne [string]$expected.sha256) {
            throw "extracted runtime file differs from the qualified manifest"
        }
    }
}

function Set-InstalledFilePermissions([string]$Directory) {
    foreach ($file in Get-ChildItem -LiteralPath $Directory -File) {
        if ([OperatingSystem]::IsWindows()) {
            $file.Attributes = $file.Attributes -bor [IO.FileAttributes]::ReadOnly
        }
        else {
            [IO.File]::SetUnixFileMode($file.FullName,
                [IO.UnixFileMode]::UserRead -bor [IO.UnixFileMode]::GroupRead -bor [IO.UnixFileMode]::OtherRead)
        }
    }
}

function Initialize-NativeSync {
    if (-not ('ImpossibleOcr.NativeSync' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace ImpossibleOcr {
  public static class NativeSync {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern bool MoveFileExW(string oldName, string newName, uint flags);
    [DllImport("libc", SetLastError=true)] static extern int open(string p, int f);
    [DllImport("libc", SetLastError=true)] static extern int fsync(int f);
    [DllImport("libc", SetLastError=true)] static extern int close(int f);
    public static void Directory(string path) {
      if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
        int fd = open(path, 0x10000);
        if (fd < 0) throw new Win32Exception(Marshal.GetLastWin32Error());
        try { if (fsync(fd) != 0) throw new Win32Exception(Marshal.GetLastWin32Error()); }
        finally { close(fd); }
      } else { throw new PlatformNotSupportedException(); }
    }
    public static void PromoteDirectory(string source, string destination) {
      if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
        if (!MoveFileExW(source, destination, 0x8)) throw new Win32Exception(Marshal.GetLastWin32Error());
      } else {
        System.IO.Directory.Move(source, destination);
      }
    }
  }
}
'@
    }
}

function Sync-Directory([string]$Directory) {
    Initialize-NativeSync
    [ImpossibleOcr.NativeSync]::Directory($Directory)
}

function Get-InstallDirectory($Manifest, $PlatformRecord, [string]$Root) {
    return [IO.Path]::Combine($Root, 'onnxruntime', [string]$Manifest.runtime.version,
        [string]$PlatformRecord.id, [string]$PlatformRecord.archive.sha256)
}

function Move-DurableDirectory([string]$Source, [string]$Destination) {
    # Load the native helper without asking Windows to flush a directory handle; MoveFileEx with
    # WRITE_THROUGH is the supported durable-rename primitive there.
    Initialize-NativeSync
    [ImpossibleOcr.NativeSync]::PromoteDirectory($Source, $Destination)
    if ([OperatingSystem]::IsLinux()) { Sync-Directory (Split-Path -Parent $Destination) }
}

function Test-InstalledRuntime($PlatformRecord, [string]$Directory) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { throw "qualified runtime is not installed" }
    foreach ($file in $PlatformRecord.requiredFiles) {
        $path = Join-Path $Directory ([string]$file.installName)
        $item = Get-Item -LiteralPath $path -Force
        if ($item.PSIsContainer -or $item.LinkType -or $item.Length -ne [long]$file.bytes -or
            (Get-Sha256 $path) -cne [string]$file.sha256) { throw "installed runtime integrity check failed" }
    }
    return $Directory
}

function Invoke-OrtPackaging {
    $manifestFile = (Resolve-Path -LiteralPath $ManifestPath).Path
    $script:Manifest = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json -Depth 32
    $requireQualified = $Action -ne 'Inspect'
    $platformRecord = Assert-Manifest $script:Manifest $Platform $requireQualified

    if ($Action -ceq 'Verify') {
        if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) { throw "Verify requires -RuntimeRoot" }
        $root = [IO.Path]::GetFullPath($RuntimeRoot)
        return Test-InstalledRuntime $platformRecord (Get-InstallDirectory $script:Manifest $platformRecord $root)
    }
    if ($Action -ceq 'Inspect' -and [string]::IsNullOrWhiteSpace($ArchivePath)) { throw "Inspect requires -ArchivePath" }
    if ($Action -ceq 'Inspect' -and $AllowDownload) { throw "Inspect never downloads" }
    if ($Action -ceq 'Install' -and [string]::IsNullOrWhiteSpace($RuntimeRoot)) { throw "Install requires -RuntimeRoot" }

    $downloaded = $false
    if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
        if (-not $AllowDownload) { throw "a local archive or explicit -AllowDownload is required" }
        $ArchivePath = Join-Path ([IO.Path]::GetTempPath()) ("impossible-ocr-ort-" + [Guid]::NewGuid().ToString('N') + '.partial')
        $downloaded = $true
        try { Receive-PinnedArchive $platformRecord $ArchivePath }
        catch {
            if (Test-Path -LiteralPath $ArchivePath) { Remove-Item -LiteralPath $ArchivePath -Force }
            throw
        }
    }
    try {
        $stageParent = if ($Action -ceq 'Install') { [IO.Path]::GetFullPath($RuntimeRoot) } else { [IO.Path]::GetTempPath() }
        [IO.Directory]::CreateDirectory($stageParent) | Out-Null
        $stageRoot = Join-Path $stageParent ('.ort-stage-' + [Guid]::NewGuid().ToString('N'))
        $staging = Join-Path $stageRoot 'payload'
        [IO.Directory]::CreateDirectory($staging) | Out-Null
        try {
            # Snapshot the caller-controlled archive into private staging while verifying it. All
            # parsing uses this exact immutable-by-access-control copy, closing a path-swap window.
            $verifiedArchive = Copy-PinnedArchive $platformRecord $ArchivePath (Join-Path $stageRoot 'source.archive')
            $derived = @(Expand-SafeArchive $platformRecord $verifiedArchive $staging)
            if ($Action -ceq 'Inspect') { return $derived | ConvertTo-Json -Depth 4 }
            Assert-DerivedFiles $platformRecord $derived
            Remove-Item -LiteralPath $verifiedArchive -Force
            Set-InstalledFilePermissions $staging
            if ([OperatingSystem]::IsLinux()) { Sync-Directory $staging }
            $destination = Get-InstallDirectory $script:Manifest $platformRecord ([IO.Path]::GetFullPath($RuntimeRoot))
            $parent = Split-Path -Parent $destination
            [IO.Directory]::CreateDirectory($parent) | Out-Null
            if (Test-Path -LiteralPath $destination) {
                $null = Test-InstalledRuntime $platformRecord $destination
                return $destination
            }
            Move-DurableDirectory $staging $destination
            $staging = $null
            return Test-InstalledRuntime $platformRecord $destination
        }
        finally { if (Test-Path -LiteralPath $stageRoot) { Remove-Item -LiteralPath $stageRoot -Recurse -Force } }
    }
    finally { if ($downloaded -and (Test-Path -LiteralPath $ArchivePath)) { Remove-Item -LiteralPath $ArchivePath -Force } }
}

if ($MyInvocation.InvocationName -ne '.') {
    Invoke-OrtPackaging
}
