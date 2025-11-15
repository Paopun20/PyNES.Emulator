Get-ChildItem app -Recurse -Filter *.py -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        python -m py_compile $_.FullName
        echo "Compiling file: $($_.FullName)"
    } catch {
        echo "Compilation failed for file: $($_.FullName)"
    }
}