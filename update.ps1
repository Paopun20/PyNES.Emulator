pip install --upgrade pkg 2>$null
pip install --upgrade pip 2>$null
pip install --upgrade setuptools 2>$null
pip install --upgrade wheel 2>$null
pip freeze | % { pip install --upgrade ($_ -split '==')[0] 2>$null }
Get-ChildItem app -Recurse -Filter *.py | ForEach-Object {
    try {
        python -m py_compile $_.FullName 2>$null
        Write-Output "OK: $($_.FullName)"
    } catch {
        Write-Output "FAIL: $($_.FullName)"
    }
}