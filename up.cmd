@echo off
pip install --upgrade pkg
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
(pip install --upgrade -r .\requirements.txt && pip freeze > requirements.txt) > nul 2>&1