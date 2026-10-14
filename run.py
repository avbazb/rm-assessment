"""Restore the checked-in model files to the source layout and start the app."""
from pathlib import Path
import hashlib, json, os, runpy, shutil, sys

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare_models():
    manifest=json.loads((ROOT/'models/manifest.json').read_text('utf8'))
    for model in manifest['files']:
        if not model.get('runtime_target'):continue
        source=(ROOT/model['path']).resolve();target=(ROOT/model['runtime_target']).resolve()
        if not source.is_relative_to(ROOT) or not target.is_relative_to(ROOT):
            raise RuntimeError('模型清单路径无效')
        if not source.is_file() or source.stat().st_size!=model['bytes']:
            raise RuntimeError('模型文件未完整下载，请重新下载或克隆完整仓库：'+model['path'])
        if sha(source)!=model['sha256']:raise RuntimeError('模型校验失败：'+model['path'])
        if target.exists():
            if target.stat().st_size!=model['bytes'] or sha(target)!=model['sha256']:
                raise RuntimeError('运行目录中存在不同版本的模型：'+str(target.relative_to(ROOT)))
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
    # Uploaded media are supplied at runtime; no example dataset is published.
    samples=ROOT/'src/preview/app/assets/samples'
    samples.mkdir(parents=True,exist_ok=True)
    catalog=samples/'catalog.json'
    if not catalog.exists():catalog.write_text('[]',encoding='utf8')

def main():
    prepare_models()
    app=ROOT/'src/preview'
    sys.path.insert(0,str(app))
    runpy.run_path(str(app/'launcher.py'),run_name='__main__')

if __name__=='__main__':main()
