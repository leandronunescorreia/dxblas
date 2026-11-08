$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rootPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$formatter = Get-Command clang-format -ErrorAction SilentlyContinue
if (-not $formatter) {
    throw @'
clang-format was not found on PATH. Install LLVM/clang-format, then rerun:
  powershell -ExecutionPolicy Bypass -File tools/format-cpp.ps1
No source files were changed.
'@
}

$extensions = @('.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.inl')
$excludedDirectories = @('.git', 'build', 'build-debug', 'build-release', 'out', 'cmake-build-debug', 'cmake-build-release')
$excludedPattern = '(^|[\\/])(' + (($excludedDirectories | ForEach-Object { [Regex]::Escape($_) }) -join '|') + ')([\\/]|$)'
$files = Get-ChildItem -Path $repoRoot -Recurse -File | Where-Object {
    $file = $_
    # Substring is compatible with Windows PowerShell/.NET Framework, unlike Path.GetRelativePath.
    $relativePath = $file.FullName.Substring($rootPrefix.Length)
    ($extensions -contains $file.Extension.ToLowerInvariant()) -and ($relativePath -notmatch $excludedPattern)
}

if (-not $files) {
    Write-Host 'No C/C++ source files found.'
    exit 0
}

foreach ($file in $files) {
    & $formatter.Source -i $file.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "clang-format failed for $($file.FullName) with exit code $LASTEXITCODE."
    }
}

Write-Host "Formatted $($files.Count) C/C++ source and header files using .clang-format."
