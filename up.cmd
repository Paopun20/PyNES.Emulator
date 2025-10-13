@echo off
(pip install --upgrade -r .\requirements.txt && pip freeze > requirements.txt) > nul 2>&1