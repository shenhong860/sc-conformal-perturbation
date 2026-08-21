$ErrorActionPreference = "Continue"
$bin = "D:\ruanjian\Latex\miktex\bin\x64"
$env:Path = "$bin;$env:Path"
$root = "D:\guo\CW1OT"
$log = Join-Path $root "results\compile_check.log"
Set-Location $root

"=== compile check $(Get-Date -Format o) ===" | Out-File $log
& "$bin\initexmf.exe" --enable-installer *>> $log
& "$bin\mpm.exe" --install=booktabs --install=cite *>> $log
& "$bin\pdflatex.exe" --enable-installer -interaction=nonstopmode -halt-on-error -synctex=1 UQ_manuscript.tex *>> $log
$c1 = $LASTEXITCODE
& "$bin\bibtex.exe" UQ_manuscript *>> $log
$c2 = $LASTEXITCODE
& "$bin\pdflatex.exe" --enable-installer -interaction=nonstopmode -halt-on-error UQ_manuscript.tex *>> $log
& "$bin\pdflatex.exe" --enable-installer -interaction=nonstopmode -halt-on-error UQ_manuscript.tex *>> $log
$c3 = $LASTEXITCODE

"EXITCODES pdflatex1=$c1 bibtex=$c2 pdflatex3=$c3" | Out-File $log -Append
"PDF exists: $(Test-Path (Join-Path $root 'UQ_manuscript.pdf'))" | Out-File $log -Append
