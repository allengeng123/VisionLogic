"""Fetch the exact torchvision checkpoint files used by VisionLogic."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

WEIGHTS = {
    'vit': 'vit_b_16-c867db91.pth',
    'resnet': 'resnet50-11ad3fa6.pth',
    'convnext': 'convnext_base-6075fbad.pth',
    'swin': 'swin_t-704ceda3.pth',
}


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=WEIGHTS, default=list(WEIGHTS))
    parser.add_argument('--output', type=Path, default=Path('data/weights'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        name = WEIGHTS[model]
        dest = args.output / name
        expected = name.rsplit('-', 1)[1].split('.')[0]
        if dest.exists():
            if not checksum(dest).startswith(expected):
                raise ValueError(f'Checksum mismatch: {dest}; move the invalid file before retrying')
            print(f'Verified {dest}')
            continue
        temp = dest.with_suffix('.download')
        with urlopen('https://download.pytorch.org/models/' + name, timeout=60) as response, temp.open('wb') as stream:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
        if not checksum(temp).startswith(expected):
            raise ValueError(f'Checksum mismatch: {temp}')
        temp.replace(dest)
        print(f'Downloaded and verified {dest}')


if __name__ == '__main__':
    main()
