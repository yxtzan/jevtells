import json

def run(clip,out,force=False):
 p=out/'shots.json'; d=[{'t0':0.0,'t1':999999.0,'label':'target'}];json.dump(d,open(p,'w'));return d
