$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scanner = "scripts/privacy-scan.ps1"
$binaryAllowlist = [ordered]@{
    "docs/assets/impossible-ocr-header.png" = "527a01bd6012b903d71046ffca627b942749a02728e704bba7bd32e7c9af2afe"
}
$binaryExtensions = @(
    ".7z", ".a", ".bmp", ".dll", ".dylib", ".exe", ".gif", ".gz", ".ico", ".jpeg",
    ".jpg", ".lib", ".onnx", ".ort", ".pdf", ".png", ".so", ".tar", ".tiff", ".webp", ".zip"
)

$patterns = [ordered]@{
    private_home = '(?i)(?:[a-z]:[\\/](?:users|documents and settings)[\\/]|/users/[^/\s"'']+/|/home/[^/\s"'']+/)'
    private_key = '-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----'
    hosted_token = '(?i)\b(?:gh[pousr]_[a-z0-9]{30,}|github_pat_[a-z0-9_]{40,}|sk-(?:proj-)?[a-z0-9_-]{20,}|hf_[a-z0-9]{30,})\b'
    agent_attribution = '(?i)(?:co-authored-by:.*(?:\b(?:bot|ai|llm|agent|codex|chatgpt|claude|openai|gemini|llama|mistral|copilot|cursor)\b|\bgpt(?:[- ]?[0-9][a-z0-9.-]*)?\b)|(?:generated|authored|written|created|built|made|assisted|implemented|reviewed)\s+(?:by|with|using)\s+(?:an?\s+)?(?:\b(?:ai|llm|agent|codex|chatgpt|claude|openai|gemini|llama|mistral|copilot|cursor)\b|\bgpt(?:[- ]?[0-9][a-z0-9.-]*)?\b)|(?:ai|llm)[- ]generated)'
}
$credentialName = '(?:[a-z0-9]+[._-])*(?:password|passwd|secret|client[_-]?secret|private[_-]?key|api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token)'
$credentialPatterns = @(
    "(?im)(?:^|[;,{])\s*(?:(?:let|const|static|var|val)\s+)?$credentialName\s*(?::\s*[^=,;]+)?\s*(?::(?!:)|=(?!>))\s*(?:`"(?<double>[^`"\r\n]*)`"|'(?<single>[^'\r\n]*)'|(?<bare>[^\s,;}#]+))",
    "(?im)(?:^|[,{])\s*(?:`"$credentialName`"|'$credentialName')\s*:\s*(?:`"(?<double>[^`"\r\n]*)`"|'(?<single>[^'\r\n]*)'|(?<bare>[^\s,;}#]+))"
)
$safeValue = '(?i)^(?:\$\{[A-Z_][A-Z0-9_]*\}|\$env:[A-Z_][A-Z0-9_]*|<[^>]+>|example|placeholder|dummy|fixture|test|test[-_]?placeholder|change[-_]?me|none|null|redacted|not[-_]?set|x{3,})$'

Push-Location $repoRoot
try {
    $files = @(git ls-files --cached --others --exclude-standard 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "unable to enumerate repository candidates" }
    $markers = @(
        (git config --get user.name 2>$null),
        (git config --get user.email 2>$null),
        $env:COMPUTERNAME,
        $env:USERPROFILE,
        $env:HOME
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) -and ([string]$_).Length -ge 4 }
    $findings = [Collections.Generic.List[string]]::new()
    foreach ($file in $files) {
        $normalized = $file.Replace('\', '/')
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { continue }
        $bytes = [IO.File]::ReadAllBytes((Resolve-Path $file))
        $extension = [IO.Path]::GetExtension($normalized).ToLowerInvariant()
        $isBinary = [Array]::IndexOf($bytes, [byte]0) -ge 0 -or $binaryExtensions -ccontains $extension
        if ($isBinary) {
            if ($binaryAllowlist.Contains($normalized)) {
                $digest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
                if ($digest -cne $binaryAllowlist[$normalized]) {
                    $findings.Add("${normalized}: reviewed binary digest mismatch")
                }
            } else {
                $findings.Add("${normalized}: unexpected binary content")
            }
            continue
        }
        $content = [Text.Encoding]::UTF8.GetString($bytes)
        if ($normalized -eq $scanner) {
            # Scan this file too. Only canonical pattern-definition lines with exact reviewed
            # hashes are neutralized. A changed definition is rejected instead of being hidden.
            $selfPatternLines = @(
                @{ selector = 'private_home'; pattern = '(?m)^[ \t]*private_home[ \t]*=.*$'; hash = '1df7006d39b146ee20767c03b9f46e119b14214fad5c68c857d12bbe220a5472' },
                @{ selector = 'private_key'; pattern = '(?m)^[ \t]*private_key[ \t]*=.*$'; hash = '79a20a59202a1d96d441ccffd65761c4d8a1d2bda05a5d237ff712914c18c49a' },
                @{ selector = 'hosted_token'; pattern = '(?m)^[ \t]*hosted_token[ \t]*=.*$'; hash = 'c826c0e23e3cc43f96bdb1a58ea1faf4f9c52cfe0d9ed0a5a7e97689abc5082f' },
                @{ selector = 'agent_attribution'; pattern = '(?m)^[ \t]*agent_attribution[ \t]*=.*$'; hash = '4bea114ffb0fca80c22d7257e968f46b0a983a2324b667d1af99a0e82e6f3bb4' },
                @{ selector = '$credentialName'; pattern = '(?m)^\$credentialName[ \t]*=.*$'; hash = '442419d445c7f472a2c6a66bf1ff5d2ca7c409b74dd68ea005d8aac15f3095e0' },
                @{ selector = '$safeValue'; pattern = '(?m)^\$safeValue[ \t]*=.*$'; hash = '6df7d35c4dd4dde33517a9215ae185ceb8a2c98f58079ed29fb5795277d201e4' }
            )
            foreach ($definition in $selfPatternLines) {
                $matches = [regex]::Matches($content, $definition.pattern)
                if ($matches.Count -ne 1) {
                    $findings.Add("${normalized}: scanner pattern definition is not canonical")
                    continue
                }
                $match = $matches[0]
                $line = $match.Value.TrimEnd("`r")
                $hash = [Convert]::ToHexString(
                    [Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($line))
                ).ToLowerInvariant()
                if ($hash -cne $definition.hash) {
                    $findings.Add("${normalized}: scanner pattern definition is not canonical")
                    continue
                }
                $replacement = "$($definition.selector) = 'placeholder'"
                $content = $content.Remove($match.Index, $match.Length).Insert($match.Index, $replacement)
            }
        }
        foreach ($entry in $patterns.GetEnumerator()) {
            if ($content -match $entry.Value) { $findings.Add("${normalized}: prohibited $($entry.Key.Replace('_', ' '))") }
        }
        foreach ($credentialPattern in $credentialPatterns) {
            foreach ($match in [regex]::Matches($content, $credentialPattern)) {
                $value = @('double', 'single', 'bare') | ForEach-Object { $match.Groups[$_].Value } | Where-Object { $_ -ne '' } | Select-Object -First 1
                if ([string]$value -notmatch $safeValue) {
                    $findings.Add("${normalized}: prohibited literal credential assignment")
                }
            }
        }
        foreach ($marker in $markers) {
            if ($content.IndexOf(([string]$marker).Trim(), [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $findings.Add("${normalized}: prohibited configured identity or machine marker")
                break
            }
        }
    }
    if ($findings.Count -gt 0) {
        $findings | Sort-Object -Unique | Write-Error
        exit 1
    }
    Write-Output "Privacy scan passed for $($files.Count) tracked and nonignored untracked candidates."
} finally {
    Pop-Location
}
