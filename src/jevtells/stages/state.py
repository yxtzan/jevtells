import json
from ..schemas import State

def run(windows,voice,scene,speaker,out,force=False):
 p=out/'states.json'; d={}
 for w in windows:d[w['id']]=State(scene=scene,speaker=speaker,subtitle={'current':w['subtitle'],'previous':w['prev_subtitle']},voice='voice features unavailable',measured_actions=['left wrist: motion statistics unavailable']).model_dump()
 json.dump(d,open(p,'w'),ensure_ascii=False,indent=2);return d
