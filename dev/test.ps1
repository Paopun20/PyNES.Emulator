$code = @"
./env/Scripts/Activate.ps1
python TUITEST.py
"@
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -Command $code" -WindowStyle Normal