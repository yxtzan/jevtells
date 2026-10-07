"""Frame-level normalized lip opening; missing/small faces stay unobserved."""
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..resources import asset_path


def read_mouths(clip, poses, out, config, force=False):
    destination=Path(out)/'mouth_features.npz';settings=config['two_person'];signature=str(sorted(settings.items()))
    if destination.exists() and not force:
        with np.load(destination,allow_pickle=False) as z:
            if str(z['signature'])==signature and len(z['opening'])==len(poses): return z['opening']
    import mediapipe as mp
    options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path(Path('models')/config['models']['face_landmarker'])),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=config['detection']['max_people'])
    result=np.full(poses.shape[:2],np.nan,np.float32)
    capture=cv2.VideoCapture(str(clip));fps=capture.get(cv2.CAP_PROP_FPS)
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            for t,candidates in enumerate(poses):
                ok,frame=capture.read()
                if not ok: raise RuntimeError('mouth decoding ended early')
                h,w=frame.shape[:2]
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                detected=detector.detect_for_video(image,round(1000*t/fps)).face_landmarks
                landmarks=[np.array([[p.x*w,p.y*h] for p in face]) for face in detected]
                costs=np.full((len(candidates),len(landmarks)),1e6)
                for p,pose in enumerate(candidates):
                    if not np.isfinite(pose[0,:2]).all(): continue
                    for f,face in enumerate(landmarks):
                        costs[p,f]=np.linalg.norm(pose[0,:2]*[w,h]-face[1])/max(np.ptp(face[:,1]),1)
                if not len(landmarks): continue
                rows,cols=linear_sum_assignment(costs)
                for p,f in zip(rows,cols):
                    face=landmarks[f];scale=np.ptp(face[:,1])
                    if costs[p,f]>settings['face_pose_distance'] or min(np.ptp(face[:,0]),scale)<settings['mouth_min_pixels']: continue
                    result[t,p]=np.linalg.norm(face[13]-face[14])/max(scale,1)
    finally: capture.release()
    np.savez(destination,opening=result,signature=signature)
    return result
