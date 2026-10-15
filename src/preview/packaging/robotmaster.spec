"""Fast preview: one ONNX pass and image-only ROI lamps; no reference NPZ."""
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

root=Path(SPECPATH).parent
backend=root/'app/backend'
assert (root/'app/assets/models/recommended.onnx').is_file()
datas=[(str(root/'app'/name),'app/'+name) for name in ('web','assets')]
for distribution in ('numpy','opencv-python-headless','onnxruntime'):
    datas+=copy_metadata(distribution)
binaries=collect_dynamic_libs('onnxruntime')
system=Path(os.environ['SystemRoot'])/'System32'
for name in ('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','msvcp140_atomic_wait.dll','msvcp140_codecvt_ids.dll','vcruntime140.dll','vcruntime140_1.dll','concrt140.dll'):
    if (system/name).is_file():binaries.append((str(system/name),'.'))
hiddenimports=['runtime.onnx_detector','runtime.detection_pipeline','runtime.fast_classical','runtime.fast_classical_v2']
for package in ('astra_competition','astra_vision','astra_recovery'):
    hiddenimports.extend(package+'.'+p.stem for p in (backend/package).glob('*.py') if p.stem!='__init__')
a=Analysis([str(root/'launcher.py')],pathex=[str(root),str(backend)],binaries=binaries,datas=datas,
    hiddenimports=hiddenimports,hookspath=[],hooksconfig={},runtime_hooks=[],
    excludes=['classical','runtime.latest_fusion','torch','torchvision','torchaudio','ultralytics','onnx','tensorflow','matplotlib','pandas','scipy','IPython','pytest','tkinter'],
    noarchive=False,optimize=1)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='robotmaster',version=str(root/'packaging/robotmaster_version.txt'),debug=False,bootloader_ignore_signals=False,
        strip=False,upx=False,console=False,disable_windowed_traceback=False)
