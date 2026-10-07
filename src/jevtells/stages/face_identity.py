"""YuNet/SFace features associated with pose heads by exclusive assignment."""
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..resources import asset_path


def associate_faces(poses, faces, width, height, settings):
    """Assign each face once; reject a face distant from the pose nose."""
    if faces is None or not len(faces):
        return {}
    costs = np.full((len(poses), len(faces)), 1e6)
    for p, pose in enumerate(poses):
        if not np.isfinite(pose[0,:2]).all(): continue
        nose = pose[0,:2]*[width,height]
        for f, face in enumerate(faces):
            centre = face[8:10]  # YuNet nose landmark
            scale = max(face[2], face[3], 1)
            costs[p,f] = np.linalg.norm(nose-centre)/scale
    rows, cols = linear_sum_assignment(costs)
    return {int(p):int(f) for p,f in zip(rows,cols) if costs[p,f] <= settings['face_pose_distance']}


def read_features(clip, poses, out, config, force=False):
    path=Path(out)/'face_features.npz'
    settings=config['two_person']
    signature=str(sorted(settings.items()))
    if path.exists() and not force:
        with np.load(path,allow_pickle=False) as z:
            if str(z['signature'])==signature and len(z['embeddings'])==len(poses):
                return {k:z[k] for k in z.files}
    models=config['models']
    detector=cv2.FaceDetectorYN.create(str(asset_path(Path('models')/models['face_detector'])), '', (320,320), settings['face_score'])
    recognizer=cv2.FaceRecognizerSF.create(str(asset_path(Path('models')/models['face_recognizer'])), '')
    embeddings=np.full((*poses.shape[:2],128),np.nan,np.float32)
    boxes=np.full((*poses.shape[:2],4),np.nan,np.float32)
    capture=cv2.VideoCapture(str(clip))
    try:
        for t,candidates in enumerate(poses):
            ok,frame=capture.read()
            if not ok: raise RuntimeError('face decoding ended early')
            h,w=frame.shape[:2];detector.setInputSize((w,h))
            _,faces=detector.detect(frame)
            for p,f in associate_faces(candidates,faces,w,h,settings).items():
                face=faces[f];boxes[t,p]=face[:4]
                if min(face[2:4]) < settings['face_min_pixels']: continue
                feature=recognizer.feature(recognizer.alignCrop(frame,face)).flatten()
                embeddings[t,p]=feature/max(np.linalg.norm(feature),1e-9)
    finally: capture.release()
    result={'embeddings':embeddings,'boxes':boxes,'signature':signature}
    np.savez(path,**result)
    return result


def anchor_references(poses, embeddings, anchors, cuts, fps, width, height, config):
    from .track import _config, select_anchor, _track_segment
    rules=_config(config);refs=[]
    for number,seconds in anchors:
        start=round(seconds*fps)
        if not 0<=start<len(poses): raise ValueError('person anchor lies outside clip')
        index=select_anchor(poses,start,number,width,height,rules.visibility_threshold)
        if index<0: raise ValueError(f'person {number} absent at anchor {seconds}')
        end=min(len(poses),start+round(fps),next((c for c in cuts if c>start),len(poses)))
        indices=_track_segment(poses,start,end-1,index,width,height,rules)
        refs.extend(embeddings[start+i,p] for i,p in enumerate(indices) if p>=0 and np.isfinite(embeddings[start+i,p]).all())
    return refs


def cosine(feature, refs):
    return max((float(np.dot(feature,ref)) for ref in refs),default=-1.) if np.isfinite(feature).all() else -1.
