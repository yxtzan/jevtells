from __future__ import annotations
import json
from ..schemas import Window

def run(transcript, shots, out, force=False):
    p = out / 'windows.json'
    if p.exists() and not force: return json.load(open(p))
    seg = transcript.get('segments', []); ws=[]; idx=0; prev=''
    for s in seg:
        start=float(s['t0']); end=float(s['t1']); words=s.get('words') or []
        while end-start > 5:
            cut=max((float(w['t1']) for w in words if start < float(w['t1']) <= start+5), default=start+5)
            text=' '.join(str(w.get('w','')) for w in words if start <= float(w['t0']) and float(w['t1']) <= cut) or str(s.get('text',''))
            ws.append(Window(id=f'W{idx:02d}', index=idx, t0=start, t1=cut, subtitle=text, prev_subtitle=prev).model_dump())
            idx+=1; prev=text; start=cut
        text=str(s.get('text',''))
        ws.append(Window(id=f'W{idx:02d}', index=idx, t0=start, t1=end, subtitle=text, prev_subtitle=prev).model_dump()); idx+=1; prev=text
    if not ws:
        duration=max((float(x['t1']) for x in shots), default=0.0)
        if duration: ws=[Window(id='W00', index=0, t0=0, t1=duration, subtitle='', prev_subtitle='', kind='silence').model_dump()]
    json.dump(ws, open(p,'w'), ensure_ascii=False, indent=2); return ws
