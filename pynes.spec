# -*- mode: python ; coding: utf-8 -*-
import os
import re
import glob
import sys
import platform
from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE

block_cipher = None

CURRENT_SYSTEM = platform.system()  # "Windows", "Linux", "Darwin"


def marker_matches(marker: str) -> bool:
    """
    Very small PEP 508 marker evaluator for:
        platform_system == "Windows"
        platform_system == "Linux"
        platform_system == "Darwin"
    """
    m = marker.replace(" ", "")
    if m.startswith("platform_system=="):
        val = m.split("==", 1)[1].strip("\"'")
        return CURRENT_SYSTEM == val
    return True  # unsupported marker → assume True


req_packages = []

if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # strip inline comments
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            # e.g. pywin32; platform_system=="Windows"
            if ";" in line:
                pkg, marker = line.split(";", 1)
                pkg = pkg.strip()
                marker = marker.strip()
                if not marker_matches(marker):
                    continue
            else:
                pkg = line

            # only remove version constraints if it's not a direct URL
            if "@" not in pkg:
                pkg = re.split(r"[><=~]", pkg)[0].strip()

            if pkg:
                req_packages.append(pkg)

req_packages = list(set(req_packages))

datas_all = []
binaries_all = []
hidden_all = []

for pkg in req_packages:
    try:
        d, b, h = collect_all(pkg)
        datas_all += d
        binaries_all += b
        hidden_all += h
    except Exception:
        # some packages like maturin/mypy have no importable data
        continue

datas_pynes = [(f, "pynes/" + os.path.basename(f)) for f in glob.glob("app/pynes/*")]
datas_all += datas_pynes

a = Analysis(
    ["app/main.py"],
    pathex=["app"],
    binaries=binaries_all,
    datas=datas_all,
    hiddenimports=hidden_all,
    excludes=["unittest","test", "pydoc", "mypy", "mypy-extensions"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=1,  # safer for numpy
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    icon="icon.ico",
)
