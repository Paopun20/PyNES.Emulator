Get-ChildItem app -Recurse -Filter *.py | ForEach-Object {
    try {
        python -m py_compile $_.FullName 2>$null
        Write-Output "OK: $($_.FullName)"
    } catch {
        Write-Output "FAIL: $($_.FullName)"
    }
}
