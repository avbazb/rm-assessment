"""Check the complete native RMUC26 map against the frozen runtime manifest."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET


def check(model):
    model = Path(model).expanduser().resolve()
    manifest_path = Path(__file__).resolve().parents[2] / 'models/rmuc2026_v20_native/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not (model / 'model.sdf').is_file():
        raise ValueError('Set RMUC_MODEL_DIR to the native rmuc2026_v20_native directory containing model.sdf, model.config and meshes.')
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid map manifest path')
        path = model / relative
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Missing or wrong-size map file: {relative}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Map hash mismatch: {relative}')
    root = ET.parse(model / 'model.sdf').getroot()
    links = root.findall('.//link')
    meshes = {node.text for node in root.findall('.//mesh/uri')}
    if len(links) != 446 or len(meshes) != 428:
        raise ValueError('Expected all 446 native CAD placements and 428 unique meshes')
    for uri in meshes:
        if not uri or not uri.startswith('model://rmuc2026_v20_native/'):
            raise ValueError(f'Unexpected mesh URI: {uri}')
        if not (model.parent / uri.removeprefix('model://')).is_file():
            raise ValueError(f'Missing mesh URI: {uri}')
    return {'map_files': len(manifest['files']), 'links': len(links), 'meshes': len(meshes), 'status': 'PASS'}


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            raise ValueError('Usage: python3 check_map.py /path/to/rmuc2026_v20_native')
        print(json.dumps(check(sys.argv[1])))
    except (ValueError, OSError, KeyError, ET.ParseError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
