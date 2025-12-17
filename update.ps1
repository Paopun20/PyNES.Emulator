uv pip install --upgrade pip setuptools wheel 2>$null

$packages = uv pip freeze | ForEach-Object { ($_ -split '==')[0] } | Where-Object { $_ }
if ($packages) {
    uv pip install --upgrade $packages
}

Get-ChildItem -Path app -Recurse -Filter *.py | ForEach-Object {
    $file = $_.FullName
    try {
        # Compile the Python file
        & uv run python -m py_compile $file
        Write-Output "PASS: $file"
    } catch {
        Write-Output "FAIL: $file - $($_.Exception.Message)"
    }
}