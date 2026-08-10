from pathlib import Path

MAX_GITHUB_FILE_SIZE = 100 * 1024 * 1024
ignored_parts = {'.git', '__pycache__', '.ipynb_checkpoints', 'datasets'}
large_files = []

for path in Path('.').rglob('*'):
    if not path.is_file():
        continue
    if any(part in ignored_parts for part in path.parts):
        continue
    size = path.stat().st_size
    if size > MAX_GITHUB_FILE_SIZE:
        large_files.append((path, size))

if large_files:
    print('These files are over GitHub normal file limit and should not be committed:')
    for path, size in large_files:
        print(f'- {path} ({size / (1024**2):.2f} MB)')
    raise SystemExit(1)

print('OK: no tracked project file is over 100 MB.')
