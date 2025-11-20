# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE

block_cipher = None

datas_np, binaries_np, hiddenimports_np = collect_all('numpy')
datas_tk, binaries_tk, hiddenimports_tk = collect_all('tkinter')

a = Analysis(
    ['app/main.py'],
    pathex=['app'],  # Make sure PyInstaller can find your modules
    binaries=binaries_np + binaries_tk,
    datas=datas_np + datas_tk + [
        ("app/pynes", "pynes"),       # Include 'pynes' folder
        ("app/icon.ico", "."),         # Icon in root of bundle
    ],
    hiddenimports=hiddenimports_np + hiddenimports,
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
    name='pynes',
    debug=False,
    strip=True,
    upx=True,
    console=False,             # Set True for console apps
    disable_windowed_traceback=True,
    argv_emulation=True,
    icon='app/icon.ico',
)
