# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE

block_cipher = None

# --- Collect NumPy ---
datas_np, binaries_np, hiddenimports_np = collect_all('numpy')
datas_tk = collect_data_files('tkinter')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries_np,
    datas=datas_np + datas_tk + [
        ("pynes", "pynes"),  # pynes core module
        ("icon.ico", "."),   # icon file (placed in root of bundle)
    ],
    hiddenimports=hiddenimports_np + [
        "numpy._core._multiarray_umath",
        "numpy._core._multiarray_tests",
        "numpy._core._dtype_ctypes",
        "numpy._core._exceptions",
        "numpy.core._dtype",
        "numpy.core._methods",
        # Tkinter imports
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.ttk",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=1, # do try 2 if it can't run it errors by numpy.
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# Single-file executable configuration
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pynes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Icon for the executable
)