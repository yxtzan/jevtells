"""HTTP/CLI parity on the documented Tim publication range; no sample logic in UI.

Run with the local UI on port 7860. Copies caches only for the direct CLI
comparison. Sets a zero API budget for that comparison to prove cache reuse.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:7860'


def http(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE+path, data=data, headers={'Content-Type':'application/json','X-JevTells-Token':TOKEN})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


TOKEN = ''
TOKEN = http('/api/bootstrap')['token']
# Documented publication starts at source 16.3 s. All manual intervals/anchors
# below use the CLI's clip-relative clock.
PARAMS = {'input':'samples/tim_interview.mov','speaker':'Tim','start':16.3,'duration':None,
          'target':[[1,0],[1,9.3]],'face_identity':True,'lang':'zh','title':'AI 读 Tim',
          'blur':[[15,15,250,60]],'others_speaking':[[2.4667,4.9667],[7.8667,9.2667],[24.3,25.1667],[30.8,31.1667]],
          'subtitles':'off','subtitle_source':'ocr','layout':'both','reframe':'auto','output_dir':'work/M9/http'}


def wait() -> dict:
    last = ''
    while True:
        result = http('/api/jobs')
        current = result.get('stage','')
        if current != last:
            print('stage:',current,flush=True)
            last = current
        if result['status'] not in {'running','cancelling'}:
            if result['status'] != 'complete':
                print('\n'.join(result['logs'][-20:]),flush=True)
                raise RuntimeError(result['status'])
            return result
        time.sleep(1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    evidence = ROOT/'work/M9'
    evidence.mkdir(exist_ok=True)
    (evidence/'validation_params.json').write_text(json.dumps(PARAMS,ensure_ascii=False,indent=2))
    http('/api/jobs',{'params':PARAMS,'mode':'prepare'})
    prepared = wait()
    print('HTTP preparation complete',flush=True)
    (evidence/'prepare_job.json').write_text(json.dumps(prepared,ensure_ascii=False,indent=2))
    http('/api/jobs',{'params':PARAMS,'mode':'run'})
    completed = wait()
    (evidence/'http_job.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2))
    source = ROOT/completed['output']
    baseline = evidence/'cli'/source.name
    if baseline.exists():
        shutil.rmtree(baseline)
    shutil.copytree(source,baseline,ignore=shutil.ignore_patterns('output_*.mp4','debug.mp4'))
    args = http('/api/command',{'params':PARAMS})['args']
    args[args.index('-o')+1] = str(baseline/'output.mp4')
    env = dict(os.environ,JEVTELLS_HOME=str(ROOT),JEVTELLS_API_LEDGER=str(evidence/'cli_zero_api.jsonl'),JEVTELLS_API_LIMIT='0',JEVTELLS_API_RESERVE='0.05')
    with (evidence/'cli.log').open('w') as log:
        subprocess.run([sys.executable,'-u','-m','jevtells.cli',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    parity = {name:{'http':digest(source/name),'cli':digest(baseline/name),'equal':digest(source/name)==digest(baseline/name)} for name in ('windows.json','judgments.json')}
    import cv2
    import numpy as np
    for layout in ('h','v'):
        cap1=cv2.VideoCapture(str(source/f'output_{layout}.mp4'))
        cap2=cv2.VideoCapture(str(baseline/f'output_{layout}.mp4'))
        count=int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rows=[]
        for index in (0, count//4,count//2,count*3//4,count-1):
            frames=[]
            for cap in (cap1,cap2):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,image=cap.read()
                assert ok
                frames.append(image)
            delta=np.abs(frames[0].astype(float)-frames[1].astype(float))
            frame_rows.append({'frame':index,'equal':bool(np.array_equal(*frames)),'mean_delta':float(delta.mean()),'max_delta':float(delta.max())})
            cv2.imwrite(str(evidence/f'parity_{layout}_{index}.jpg'),frames[0])
        cap1.release();cap2.release()
        parity['frames_'+layout]=frame_rows
    parity['new_api_cost']=completed['cost']
    parity['cli_api_calls']=0 if not (evidence/'cli_zero_api.jsonl').exists() else len((evidence/'cli_zero_api.jsonl').read_text().splitlines())
    (evidence/'parity.json').write_text(json.dumps(parity,ensure_ascii=False,indent=2))
    assert all(parity[n]['equal'] for n in ('windows.json','judgments.json'))
    assert all(r['equal'] for n in ('h','v') for r in parity['frames_'+n])
    # Change a visual parameter and verify the HTTP task chooses --from render.
    rerender={**PARAMS,'title':'AI 读 Tim · M9'}
    result=http('/api/jobs',{'params':rerender,'mode':'run'})
    assert result['render_only']
    rendered=wait()
    (evidence/'rerender_job.json').write_text(json.dumps(rendered,ensure_ascii=False,indent=2))
    assert rendered['cost']==0 and not rendered.get('cost_unknown')
    assert not any(line.startswith('UI_STAGE ') and line not in {'UI_STAGE render'} for line in rendered['logs'])
    print('HTTP/CLI parity and render-only checks passed',flush=True)


if __name__=='__main__':
    main()
