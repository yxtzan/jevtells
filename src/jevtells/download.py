"""Download model assets and the OFL fonts used by the renderers."""

from pathlib import Path
from urllib.request import Request, urlopen
from .resources import asset_roots

ASSETS = {
    "models/face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "models/face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface_2021dec.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "models/YuNet-LICENSE.txt": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE",
    "models/SFace-LICENSE.txt": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/LICENSE",
    "models/pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "models/hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "assets/fonts/NotoSansCJKsc-Regular.otf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "assets/fonts/NotoSansCJKsc-Bold.otf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
    "assets/fonts/NotoSansCJKsc-Black.otf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Black.otf",
    "assets/fonts/NotoSerifCJKsc-Bold.otf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf",
    "assets/fonts/NotoSerifCJKsc-Black.otf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Black.otf",
    "assets/fonts/NotoSansMono-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmono/NotoSansMono%5Bwdth,wght%5D.ttf",
    "assets/fonts/NotoCJK-LICENSE.txt": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE",
    "assets/fonts/NotoSerifCJK-LICENSE.txt": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/LICENSE",
    "assets/fonts/NotoSansMono-OFL.txt": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmono/OFL.txt",
}

def download_file(relative_path: str, url: str) -> None:
    """Download one asset, or verify that an existing asset is present."""
    destination = asset_roots()[0] / relative_path
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
