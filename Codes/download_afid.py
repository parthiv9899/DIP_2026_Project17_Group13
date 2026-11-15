"""Downloads the three official AFID archives from AITEX and extracts them into ./data.

Needs py7zr (pip install py7zr) or the 7z command line tool. Skips archives already extracted.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = "https://www.aitex.es/wp-content/uploads/2019/07/"
PARTS = ("NODefect_images", "Defect_images", "Mask_images")


def main(dest):
    dest.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        if (dest / part).exists():
            print(f"{part}: already there")
            continue
        archive = dest / f"{part}.7z"
        if not archive.exists():
            print(f"downloading {BASE}{part}.7z")
            urllib.request.urlretrieve(f"{BASE}{part}.7z", archive)
        try:
            import py7zr
            with py7zr.SevenZipFile(archive) as z:
                z.extractall(dest)
        except ImportError:
            if not shutil.which("7z"):
                sys.exit("install py7zr (pip install py7zr) or 7z to extract the archives")
            subprocess.run(["7z", "x", "-y", f"-o{dest}", str(archive)], check=True)
        archive.unlink()
        print(f"{part}: extracted")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data")
