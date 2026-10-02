from pathlib import Path
import cv2,numpy as np

def run(clip,out,force=False):
 p=out/'keypoints.npz'
 if p.exists() and not force:return np.load(p)
 cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(3));h=int(cap.get(4)); poses=[]; hands=[]
 while True:
  ok,fr=cap.read()
  if not ok:break
  # deterministic placeholder landmarks; real detector can replace in M2
  pose=np.full((33,4),np.nan,np.float32); hand=np.full((2,21,3),np.nan,np.float32)
  pose[0,:2]=[.5,.3]; pose[11,:2]=[.4,.45];pose[12,:2]=[.6,.45];pose[15,:2]=[.35,.65];pose[16,:2]=[.65,.65]
  poses.append(pose);hands.append(hand)
 cap.release(); arr={'fps':fps,'width':w,'height':h,'n_frames':len(poses),'t':np.arange(len(poses))/fps,'pose':np.array(poses),'hands':np.array(hands),'pose_present':np.ones(len(poses),bool),'hand_present':np.zeros((len(poses),2),bool)};np.savez(p,**arr);return arr
