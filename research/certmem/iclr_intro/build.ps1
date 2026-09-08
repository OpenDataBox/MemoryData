param([string]$TectonicPath = '')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not $TectonicPath) {
        $bundledCompiler = Join-Path $PSScriptRoot '../../../.runtime-cache/paper-tools/tectonic/tectonic.exe'
        if (Test-Path -LiteralPath $bundledCompiler) {
            $TectonicPath = (Resolve-Path -LiteralPath $bundledCompiler).Path
        } elseif (Get-Command tectonic -ErrorAction SilentlyContinue) {
            $TectonicPath = (Get-Command tectonic).Source
        }
    }
    if ($TectonicPath) {
        & $TectonicPath --keep-logs --keep-intermediates main.tex
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic compilation failed.' }
    } else {
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'pdfLaTeX compilation failed.' }
        & bibtex main
        if ($LASTEXITCODE -ne 0) { throw 'BibTeX compilation failed.' }
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'pdfLaTeX compilation failed.' }
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'pdfLaTeX compilation failed.' }
    }
} finally { Pop-Location }
