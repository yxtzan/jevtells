"""Download M1 model/font assets; URLs can be replaced when upstream assets are pinned."""
from pathlib import Path
import urllib.request
ROOT=Path(__file__).parents[1]
ASSETS={'models/pose_landmarker_full.task':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/1/pose_landmarker_full.task','models/hand_landmarker.task':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/1/hand_landmarker.task','assets/fonts/NotoSansSC-Regular.otf':'https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'}
for rel,url in ASSETS.items():
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): urllib.request.urlretrieve(url,p);print('downloaded',rel)
 else: print('exists',rel)
