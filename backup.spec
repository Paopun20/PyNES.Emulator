from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ
from PyInstaller.building.api import EXE

# Include your own folders and the NumPy data/binaries/hiddenimports
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('pynes', 'pynes'),        # pynes core module
        ('icon', 'icon'),          # icon folder
    ],
    hiddenimports=['numpy', 'numpy.core._methods', 'numpy.lib.format'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)

exe = EXE(
    PYZ(a.pure),
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pynes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'
)
