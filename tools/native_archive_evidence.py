"""Copy a completed diagnostic run's reviewable evidence, excluding logits blobs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    result = json.loads((source / 'results.json').read_text(encoding='utf8'))
    if result.get('status') not in ('pass', 'completed', 'interrupted_between_cases', 'failed'):
        raise ValueError('run must have an explicit terminal status before archival')
    names = {'results.json', 'requests.json', 'command.json', 'engine.log',
             'http.json', 'telemetry.jsonl', 'gpu.csv'}
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.name in names)
    destination.mkdir(parents=True)
    manifest = {'source': str(source), 'status': result['status'], 'files': []}
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest['files'].append({'path': relative.as_posix(), 'bytes': path.stat().st_size,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(f'{len(files)} evidence files archived to {destination}')


if __name__ == '__main__':
    main()
