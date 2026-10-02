import json
from ..schemas import Window

def run(transcript,shots,out,force=False):
 p=out/'windows.json'
 if p.exists() and not force:return json.load(open(p))
 seg=transcript.get('segments',[]); ws=[]
 for i,s in enumerate(seg): ws.append(Window(id=f'W{i:02d}',index=i,t0=s['t0'],t1=s['t1'],subtitle=s['text'],prev_subtitle=seg[i-1]['text'] if i else '').model_dump())
 json.dump(ws,open(p,'w'),ensure_ascii=False,indent=2);return ws
