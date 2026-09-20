$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $projectDir "manuscript"
$source = Join-Path $sourceDir "main.tex"
$buildPdf = Join-Path $sourceDir "main.pdf"
$outputDir = Join-Path $projectDir "output\pdf"
$output = Join-Path $outputDir "REIM_main_four_seed.pdf"
$localEngine = Join-Path $projectDir ".tools\tectonic\tectonic.exe"

if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Missing manuscript source: $source"
}

if (Test-Path -LiteralPath $localEngine -PathType Leaf) {
    $engine = $localEngine
} else {
    $tectonic = Get-Command tectonic -ErrorAction SilentlyContinue
    if (-not $tectonic) {
        throw "Tectonic was not found. Install it or place tectonic.exe at $localEngine"
    }
    $engine = $tectonic.Source
}

Push-Location $sourceDir
try {
    & $engine --keep-logs main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $buildPdf -PathType Leaf)) {
    throw "LaTeX compilation did not produce $buildPdf"
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Copy-Item -LiteralPath $buildPdf -Destination $output -Force
Write-Output "Compiled $output"
