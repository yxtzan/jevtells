import json, cv2

def run(clip, out, force=False):
    p=out/'shots.json'
    if p.exists() and not force: return json.load(open(p))
    cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; n=cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release()
    d=[{'t0':0.0,'t1':float(n/fps),'label':'target'}]
    json.dump(d,open(p,'w')); return d
