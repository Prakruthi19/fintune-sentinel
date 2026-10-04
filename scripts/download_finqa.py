"""Download FinQA (MIT license, Chen et al. 2021) at a pinned commit and
verify checksums, so every run uses byte-identical data.

    python scripts/download_finqa.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
BASE = f"https://raw.githubusercontent.com/czyssrs/FinQA/{COMMIT}/dataset"
SHA256 = {
    "train.json": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    "dev.json": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test.json": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
}


def main(out_dir: str = "data/finqa") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        path = out / name
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE}/{name}", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            path.unlink()
            sys.exit(f"checksum mismatch for {name}: got {digest}")
        print(f"ok {name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
