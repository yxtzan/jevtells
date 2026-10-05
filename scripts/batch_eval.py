"""Run a manifest with stage caches and summarize measured tune/holdout evidence."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jevtells.cli import _clip_id
from check_layout import check
from render_review import overview


def read(directory: Path, name: str, default: Any) -> Any:
    path=directory/name
    return json.loads(path.read_text()) if path.exists() else default


def collect(directory: Path, identifier: str, group: str) -> dict[str, Any]:
    metadata=read(directory,'run_meta.json',{})
    with np.load(directory/'keypoints.npz',allow_pickle=False) as points:
        present=points['pose_present'].astype(bool)
        locked=present & (points['target_index']>=0)
        visible=points['hand_present'].astype(bool)
        lock_rate=float(locked.mean())
        hand_rate=float(visible.mean())
    actions=read(directory,'actions.json',[])
    if isinstance(actions,dict): actions=actions['events']
    narration=read(directory,'narrate_meta.json',{})
    count=len(narration.get('windows',{}))
    checks=[check(directory,kind) for kind in ('h','v')]
    (directory/'layout_check.json').write_text(json.dumps(checks,indent=2))
    rendered=read(directory,'render_meta.json',{})
    displayed=set()
    for kind in ('h','v'):
        with (directory/f'layout_trace_{kind}.jsonl').open() as handle:
            for line in handle:
                displayed.update(label['event'] for label in json.loads(line)['labels'])
    api=metadata.get('api',{})
    calls=sum(stage.get('calls',0) for name,stage in api.items() if isinstance(stage,dict))
    duration=float(metadata['duration_s'])
    return {'id':identifier,'group':group,'duration_s':duration,'target_lock_rate':lock_rate,'hand_visibility_rate':hand_rate,'windows':len(read(directory,'windows.json',[])),'events':len(actions),'events_by_type':json.dumps(dict(Counter(action['type'] for action in actions)),sort_keys=True),'displayed_labels':len(displayed),'far_shots':sum(bool(shot.get('far')) for shot in read(directory,'shots.json',[])),'api_calls':calls,'api_cost':api.get('total_cost'),'narration_retries':narration.get('retries',0),'narration_fallbacks':narration.get('fallbacks',0),'narration_fallback_ratio':narration.get('fallbacks',0)/count if count else 0,'leader_face_frames':sum(item.get('leader_face_frames',0) for item in checks),'leader_midline_frames':sum(item.get('leader_midline_frames',0) for item in checks),'hidden_label_frames':sum(item.get('hidden_label_frames',0) for item in checks),'hidden_leader_frames':sum(item.get('hidden_leader_frames',0) for item in checks),'relaxed_count':sum(item.get('relaxed_count',0) for item in checks),'forced_count':sum(item.get('forced_count',0) for item in checks),'leader_torso_frames':sum(item.get('leader_torso_frames',0) for item in checks),'suppressed_label_frames':sum(item.get('suppressed_label_frames',0) for item in checks),'blocked_face_leader_frames':sum(item.get('blocked_face_leader_frames',0) for item in checks),'layout_violation_frames':sum(item['violation_frames'] for item in checks),'layout_fallback_frames':sum(item['fallback_frames'] for item in checks),'render_h_seconds':rendered['h']['elapsed_s'],'render_v_seconds':rendered['v']['elapsed_s'],'events_per_minute':len(actions)*60/duration if duration else None,'output':str(directory),'alerts':''}


def alerts(rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    for group in ('tune','holdout'):
        selected=[row for row in rows if row['group']==group and 'error' not in row]
        rates=[row['events_per_minute'] for row in selected]
        median=statistics.median(rates) if rates else 0
        factor=float(settings.get('event_rate_factor',3))
        for row in selected:
            reasons=[]
            if row['target_lock_rate']<float(settings.get('min_lock_rate',.8)): reasons.append('target lock <80%')
            if row['narration_fallback_ratio']>float(settings.get('max_fallback_ratio',.3)): reasons.append('narration fallback >30%')
            if row['layout_violation_frames']>0: reasons.append('layout collisions')
            if row.get('output') and any(item['face_crossing_ratio']>.02 for item in read(Path(row['output']),'layout_check.json',[])): reasons.append('face crossing exceeds 2%')
            if row.get('hidden_label_frames',0) or row.get('hidden_leader_frames',0): reasons.append('incomplete labels or leaders')
            if len(rates)>=2 and median>0 and (row['events_per_minute']>median*factor or row['events_per_minute']<median/factor): reasons.append('event rate outside group median factor')
            row['alerts']='; '.join(reasons)


def write_summary(directory: Path, rows: list[dict[str, Any]]) -> None:
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (directory/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    content=['# Batch evaluation','', 'Metrics below describe the retained real API stage records, including retries. Cached stages make no new API calls. Render times are from the most recent actual encode, not cache lookup time. Each group is summarized separately; example holdout fixtures are not independent new videos.','']
    for group in ('tune','holdout'):
        content += [f'## {group}','', '| ID | seconds | lock | hands | windows | events / kinds | labels | far | API calls / cost | retries / fallback | H / V encode s | layout violations / fallback frames | face / midline crossing frames | hidden label / leader frames; relaxed / forced | alerts |','|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|---|---|']
        for row in rows:
            if row['group']!=group: continue
            if 'error' in row:
                content.append(f"| {row['id']} | BLOCKED: {row['error']} |")
                continue
            content.append(f"| {row['id']} | {row['duration_s']:.2f} | {row['target_lock_rate']:.1%} | {row['hand_visibility_rate']:.1%} | {row['windows']} | {row['events']} / {row['events_by_type']} | {row['displayed_labels']} | {row['far_shots']} | {row['api_calls']} / {row['api_cost']} | {row['narration_retries']} / {row['narration_fallbacks']} | {row['render_h_seconds']:.2f} / {row['render_v_seconds']:.2f} | {row['layout_violation_frames']} / {row['layout_fallback_frames']} | {row.get('leader_face_frames',0)} / {row.get('leader_midline_frames',0)} | {row.get('hidden_label_frames',0)} / {row.get('hidden_leader_frames',0)}; {row.get('relaxed_count',0)} / {row.get('forced_count',0)} | {row['alerts']} |")
        content.append('')
    content += ['## Overviews','']
    for row in rows:
        if 'error' not in row:
            for kind in ('h','v'):
                content.append(f"- {row['id']} {kind}: {row['id']}/overview_{kind}.png")
    (directory/'summary.md').write_text('\n'.join(content)+'\n')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cli-entry',type=Path,help='optional instrumented Python CLI entry point')
    args=parser.parse_args()
    manifest=yaml.safe_load(args.manifest.read_text())
    clips=manifest['clips'] if isinstance(manifest,dict) else manifest
    if not isinstance(clips,list) or len({clip['id'] for clip in clips})!=len(clips):
        raise ValueError('manifest needs uniquely identified clips')
    destination=args.output or ROOT/'work/_eval'/datetime.now().strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in clips:
        if item['group'] not in ('tune','holdout'): raise ValueError('group must be tune or holdout')
        source=Path(item['input']);source=source if source.is_absolute() else ROOT/source
        parameters=item.get('parameters',{})
        command=[sys.executable,str(args.cli_entry.resolve())] if args.cli_entry else [sys.executable,'-m','jevtells.cli']
        command+=['run',str(source),'--speaker',parameters.get('speaker','Speaker')]
        for key,value in parameters.items():
            if key=='speaker': continue
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value: command.append(flag)
            else:
                for entry in value if isinstance(value,list) else [value]: command += [flag,str(entry)]
        print(f"evaluating {item['id']}",flush=True)
        with (destination/f"{item['id']}.log").open('w') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        out=ROOT/'work'/_clip_id(source,float(parameters.get('start',0)),parameters.get('duration'))
        try:
            if process.returncode: raise RuntimeError(f'CLI exit {process.returncode}; see log')
            row=collect(out,item['id'],item['group']);row['notes']=item.get('notes','')
            evidence=destination/item['id'];evidence.mkdir(exist_ok=True)
            for kind in ('h','v'): overview(out/f'output_{kind}.mp4',evidence/f'overview_{kind}.png',kind)
        except Exception as error:
            row={'id':item['id'],'group':item['group'],'error':str(error)}
        rows.append(row)
        write_summary(destination,rows)
    alerts(rows,manifest.get('anomalies',{}) if isinstance(manifest,dict) else {})
    write_summary(destination,rows)
    print(destination,flush=True)
    if any('error' in row for row in rows): raise SystemExit(1)


if __name__=='__main__': main()
