a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('pynes', 'pynes'),        # โฟลเดอร์ pynes
        ('icon', 'icon'),          # โฟลเดอร์ icon
        ('icon/icon128.ico', 'icon/icon128.ico')  # ไฟล์ icon
    ],
    hiddenimports=[],
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
    icon='icon128.ico'
)