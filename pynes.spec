# -*- mode: python ; coding: utf-8 -*-
import os
import re
import glob
from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE

block_cipher = None

req_packages = []

if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip comments / empty
            if not line or line.startswith("#"):
                continue

            # remove version markers (>=, ==, <=, ~=)
            pkg = re.split(r"[><=;~]", line)[0].strip()

            if pkg:
                req_packages.append(pkg)

# remove duplicates
req_packages = list(set(req_packages))

datas_all = []
binaries_all = []
hiddenimports_all = []

for pkg in req_packages:
    try:
        datas, bins, hidden = collect_all(pkg)
        datas_all += datas
        binaries_all += bins
        hiddenimports_all += hidden
    except Exception:
        # Some packages (maturin, mypy, typing-only libs) cannot be collected
        continue

# Include app/pynes folder
datas_pynes = [(f, "pynes/" + os.path.basename(f))
               for f in glob.glob("app/pynes/*")]

datas_all += datas_pynes
datas_all += [("app/icon.ico", ".")]

a = Analysis(
    ['app/main.py'],
    pathex=['app'],
    binaries=binaries_all,
    datas=datas_all,
    hiddenimports=hiddenimports_all,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="pynes",
    debug=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=True,
    icon="app/icon.ico",
)
