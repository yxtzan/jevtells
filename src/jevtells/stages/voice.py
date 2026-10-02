import json, numpy as np

def run(wav,out,force=False):
 p=out/'voice_features.json'
 if p.exists() and not force:return json.load(open(p))
 d={'t':[],'rms_db':[],'f0_hz':[],'baseline_rms_db':None,'baseline_f0_hz':None};json.dump(d,open(p,'w'));return d
