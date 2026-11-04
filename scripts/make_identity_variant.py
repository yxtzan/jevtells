"""Generate an actual shirt recolor edit and measure face/clothing reacquisition.

Pose geometry is unchanged by the pixel-only transform, so the source's real
pose detections are reused. Face features are measured again on the encoded
variant; the edit boundary comes from the generator, not a hand-tuned detector.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jevtells.config import load_config
from jevtells.stages import appearance,face_identity
from jevtells.cli import _load_npz


def generate(source, out, start, seconds, config):
    out.mkdir(parents=True,exist_ok=True)
    points=_load_npz(source/'keypoints.npz');fps=float(points['fps'])
    a=round(start*fps);n=round(seconds*fps)
    poses=points['pose'][a:a+n]
    if len(poses)!=n or not points['pose_present'][a:a+n].all():
        raise ValueError('Choose an excerpt with a continuously observed target')
    w,h=int(points['width']),int(points['height']);settings=config['synthetic_recolor']
    video=out/'same_person_changed_shirt.mp4'
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    capture=cv2.VideoCapture(str(source/'clip.mp4'))
    try:
        for half in range(2):
            capture.set(cv2.CAP_PROP_POS_FRAMES,a)
            for i,pose in enumerate(poses):
                ok,frame=capture.read()
                if not ok: raise RuntimeError('Source decoding failed')
                if half:
                    mask=np.zeros((h,w),np.uint8)
                    polygon=np.round(pose[[11,12,24,23],:2]*[w,h]).astype(np.int32)
                    cv2.fillConvexPoly(mask,polygon,255)
                    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                    selected=mask>0
                    hsv[:,:,0][selected]=(hsv[:,:,0][selected].astype(int)+settings['hue_shift'])%180
                    hsv[:,:,1][selected]=np.maximum(hsv[:,:,1][selected],settings['saturation_floor'])
                    changed=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                    frame[selected]=changed[selected]
                proc.stdin.write(frame.tobytes())
    finally:
        capture.release();proc.stdin.close()
    if proc.wait(): raise RuntimeError('Could not encode recolor variant')
    candidates=np.concatenate((poses,poses))[:,None]
    anchors=[(1,0.)];cuts=[n]
    descriptors=appearance.read_appearances(video,candidates,config['detection']['pose_visibility_threshold'])
    clothing,cloth_meta=appearance.track_appearance(candidates,descriptors,anchors,cuts,fps,w,h,config)
    features=face_identity.read_features(video,candidates,out,config)
    refs=face_identity.anchor_references(candidates,features['embeddings'],anchors,cuts,fps,w,h,config)
    faces,face_meta=face_identity.refine_single(candidates,features['embeddings'],refs,clothing,anchors,fps,w,h,config,cuts)
    receipt={'source':str(source),'source_start':start,'seconds_per_half':seconds,'known_edit_frame':n,'settings':settings,
             'frames_after_edit':n,'clothing_recovered_frames':int((clothing[n:]>=0).sum()),'face_recovered_frames':int((faces[n:]>=0).sum()),
             'face_reference_samples':len(refs),'face_methods':face_meta['method_counts'],'clothing_shots':cloth_meta['shots']}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2));np.savez(out/'indices.npz',clothing=clothing,faces=faces)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--seconds',type=float,default=4.)
    args=parser.parse_args()
    print(json.dumps(generate(args.source_dir,args.output,args.start,args.seconds,load_config()),indent=2))

if __name__=='__main__':main()
