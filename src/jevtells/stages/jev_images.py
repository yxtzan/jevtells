"""Target-only image extraction and UUID assets for experimental Jev input."""
from __future__ import annotations

import base64
import math
import uuid
from pathlib import Path

import cv2
import numpy as np


def compliant(image, settings):
    if image is None or not image.size:
        raise ValueError('Empty Jev image')
    h,w = image.shape[:2]
    low,high = settings['min_side'], settings['max_side']
    # Padding preserves faces/hands even for narrow crops.
    scale = min(1., settings['preferred_side']/max(w,h), high/max(w,h))
    w,h = max(1,round(w*scale)),max(1,round(h*scale))
    image = cv2.resize(image,(w,h))
    target_w,target_h = max(low,w),max(low,h)
    if target_w*target_h < settings['min_pixels']:
        target_w = max(target_w, math.ceil(settings['min_pixels']/target_h))
    return cv2.copyMakeBorder(image, (target_h-h)//2, target_h-h-(target_h-h)//2,
        (target_w-w)//2, target_w-w-(target_w-w)//2, cv2.BORDER_REPLICATE)


def asset(path: Path):
    suffix = path.suffix.lower()
    if suffix not in ('.jpg','.jpeg','.png'):
        raise ValueError('Jev accepts only JPEG/PNG')
    return str(uuid.uuid4()), {'type':'image', 'format':'png' if suffix == '.png' else 'jpg',
        'data_base64':base64.b64encode(path.read_bytes()).decode('ascii')}


def read_frame(clip, frame_index):
    capture = cv2.VideoCapture(str(clip))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)); ok,frame = capture.read()
        if not ok:
            raise RuntimeError(f'Could not decode frame {frame_index}')
        return frame
    finally:
        capture.release()


def crop_target(frame, pose, hands, settings, face_only=False):
    h,w = frame.shape[:2]; visibility = settings['visibility']
    visible = np.isfinite(pose[:,:2]).all(axis=1) & (pose[:,3] >= visibility)
    visible &= (pose[:,0]>=0)&(pose[:,0]<=1)&(pose[:,1]>=0)&(pose[:,1]<=1)
    indices = np.arange(11) if face_only else np.arange(25)
    xy = pose[indices[visible[indices]],:2] * [w,h]
    if not len(xy):
        raise ValueError('Target has no observable face/upper-body landmarks')
    if not face_only:
        for hand in hands:
            valid = np.isfinite(hand[:,:2]).all(axis=1)&(hand[:,0]>=0)&(hand[:,0]<=1)&(hand[:,1]>=0)&(hand[:,1]<=1)
            xy = np.concatenate((xy,hand[valid,:2]*[w,h]))
    lo,hi = xy.min(axis=0),xy.max(axis=0)
    pad = np.maximum(hi-lo,1)*settings['crop_padding']
    lo = np.maximum(0,np.floor(lo-pad)).astype(int)
    hi = np.minimum([w,h],np.ceil(hi+pad)).astype(int)
    return compliant(frame[lo[1]:hi[1],lo[0]:hi[0]],settings), [*lo.tolist(),*hi.tolist()]


def select_frames(window, points, actions):
    fps = float(points['fps']); n = len(points['pose'])
    a,b = float(window['t0']),float(window['t1'])
    valid = np.flatnonzero((points['t']>=a)&(points['t']<b)&np.asarray(points['pose_present']))
    if not len(valid):
        return []
    events = [e for e in actions.get('events',[]) if a<=float(e.get('mid',(e['t0']+e['t1'])/2))<b]
    magnitude = {'small':0,'medium':1,'large':2}
    strongest = max(events,key=lambda e:(magnitude.get(e.get('magnitude'),0),float(e.get('t1',0))-float(e.get('t0',0))),default=None)
    if strongest:
        active = float(strongest.get('mid',(strongest['t0']+strongest['t1'])/2))
    else:
        # With no measured event, select peak observed wrist displacement.
        wrists = points['pose'][valid][:,[15,16],:2] * [float(points['width']),float(points['height'])]
        velocity = np.linalg.norm(np.diff(wrists,axis=0),axis=-1)
        scores = np.max(np.where(np.isfinite(velocity),velocity,0),axis=1) if len(valid)>1 else np.array([0.])
        active = valid[min(len(valid)-1,int(np.argmax(scores))+1)]/fps
    return [int(valid[np.argmin(abs(valid/fps-t))]) for t in (a,(a+b)/2,active)]


def extract(clip, out, windows, points, actions, settings, face_only=False):
    directory = Path(out)/('faces' if face_only else 'frames'); directory.mkdir(parents=True,exist_ok=True)
    result = {}; manifest = {}
    for window in windows:
        identifier = window['id']; paths = []; records = []
        for slot,index in enumerate(select_frames(window,points,actions)):
            image,box = crop_target(read_frame(clip,index),points['pose'][index],points['hands'][index],settings,face_only)
            path = directory/f'{identifier}_{slot}.jpg'
            if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,settings['jpeg_quality']]):
                raise RuntimeError(f'Could not save {path}')
            paths.append(str(path.resolve())); records.append({'path':str(path.resolve()),'frame':index,'time':index/float(points['fps']),'crop':box,'size':[image.shape[1],image.shape[0]]})
        result[identifier] = paths; manifest[identifier] = records
    import json
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return result


def bind(state):
    """Replace local paths with references only at the request boundary."""
    value = dict(state); assets = {}; refs = []
    for path in value.get('frames',[]):
        reference,item = asset(Path(path)); assets[reference] = item; refs.append(reference)
    value['frames'] = refs
    return value,assets
