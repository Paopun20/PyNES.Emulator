# Upgrade core tools
uv pip install --upgrade pip setuptools wheel 2>$null

# Upgrade all installed packages at once
$packages = uv pip freeze | ForEach-Object { ($_ -split '==')[0] } | Where-Object { $_ }
if ($packages) {
    uv pip install --upgrade $packages
}

# Compile all Python files under 'app'
Get-ChildItem app -Recurse -Filter *.py | ForEach-Object {
    $file = $_.FullName
    try {
        # Uncomment to test error handling
        # throw "Error compiling $file"
        
        & python -m py_compile $file
        Write-Output "PASS: $file"
    } catch {
        Write-Output "FAIL: ($file) - $($_.Exception.Message)"
    }
}
