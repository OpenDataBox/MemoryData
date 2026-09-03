param([string]$TectonicPath = '')
$ErrorActionPreference = 'Stop'
$paperDirectory = $PSScriptRoot
Push-Location -LiteralPath $paperDirectory
try {
    if (-not $TectonicPath) {
        $workspaceTectonic = Join-Path $paperDirectory '../../.runtime-cache/paper-tools/tectonic/tectonic.exe'
        if (Test-Path -LiteralPath $workspaceTectonic) {
            $TectonicPath = (Resolve-Path -LiteralPath $workspaceTectonic).Path
        } elseif (Get-Command tectonic -ErrorAction SilentlyContinue) {
            $TectonicPath = (Get-Command tectonic).Source
        }
    }
    if ($TectonicPath) {
        & $TectonicPath --keep-logs --keep-intermediates main.tex
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic compilation failed.' }
    } else {
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'First pdfLaTeX pass failed.' }
        & bibtex main
        if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'Second pdfLaTeX pass failed.' }
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'Final pdfLaTeX pass failed.' }
    }
    Write-Output (Join-Path $paperDirectory 'main.pdf')
} finally {
    Pop-Location
}
