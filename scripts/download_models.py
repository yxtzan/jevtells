"""Download the MediaPipe models and the Chinese font required by M1."""

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "models/pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "models/hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "assets/fonts/NotoSansCJKsc-Regular.otf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
}

def download_file(relative_path: str, url: str) -> None:
    """Download one asset, or verify that an existing asset is present."""
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists {relative_path}")
        return
    try:
        request = Request(url, headers={"User-Agent": "jevtells/0.1"})
        with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
            handle.write(response.read())
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {relative_path}: {error}") from error
    if destination.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {relative_path}")
    print(f"downloaded {relative_path}")

def main() -> None:
    """Download every required M1 asset and fail on the first error."""
    for path, url in ASSETS.items():
        download_file(path, url)

if __name__ == "__main__":
    main()
