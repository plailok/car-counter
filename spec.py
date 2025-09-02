# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = collect_submodules('ultralytics')

a = Analysis(
    ['car_detect/app.py'],   # точка входа
    pathex=['.'],
    binaries=[],
    datas=[
        ('models/yolov8n.pt', 'models'),
        ('models/yolov8m.pt', 'models'),
        ('models/yolov8x.pt', 'models'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CarCounterApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # окно консоли не открывается
    icon='icons/car-100.ico'  # можно добавить иконку
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='CarCounterApp'
)