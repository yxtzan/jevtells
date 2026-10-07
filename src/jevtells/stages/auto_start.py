"""Locate sustained solo primary speech and align to a real cue/edit boundary."""
import json,copy
from pathlib import Path
import numpy as np
from . import prepare,pose,track,shots
from .framing import person_mask


def find_start(points,detections,cuts,windows,primary,intervals,config):
    fps=float(points['fps']);poses=detections['poses_all'];settings=config['shots'];required=max(1,round(config['two_person']['auto_start_seconds']*fps))
    count=0;candidate=None
    for i in range(len(poses)):
        # Use the same shared valid-person criteria as framing and face protection.
        people=person_mask(poses[i],{**settings,'visibility':config['detection']['pose_visibility_threshold']})
        t=i/fps;visible=points['target_index'][i]>=0 and int(people.sum())==1
        speaking=any(w['t0']<=t<w['t1'] and w.get('speaker')==primary for w in windows) if windows is not None else not any(a<=t<b for a,b in intervals)
        count=count+1 if visible and speaking else 0
        if count>=required: candidate=(i-required+1)/fps;break
    if candidate is None: raise ValueError('--start auto: no >=2s solo primary speech found; use --start SECONDS or review --speaker-map')
    boundaries=[0.,*[c/fps for c in cuts],*[w['t0'] for w in (windows or [])]]
    aligned=max((t for t in boundaries if t<=candidate+1e-6),default=candidate)
    return float(aligned),float(candidate)


def resolve(arguments,config):
    from ..cli import _run_pose,_load_npz,_clip_id,_parse_targets
    from .speakers import parse_intervals
    full=Path('work')/(Path(arguments.input).stem+'_autostart')
    clone=copy.deepcopy(arguments);clone.start=0.;clone.duration=None;clone.output=str(full/'auto.mp4');clone.until='segment'
    if len(arguments.persons)==2:
        from .interview import run
        run(clone,config)
        data=_load_npz(full/'detections.npz');points=_load_npz(full/'person_0.npz');windows=json.loads((full/'windows.json').read_text())
        primary=next(iter(arguments.persons));cut_frames=json.loads((full/'track_meta.json').read_text())['cuts']
    else:
        clip,_,_=prepare.run(Path(arguments.input),full,False,0.,None,config)
        data=_run_pose(clip,full,False,config)
        points=track.run(data,full,_parse_targets(arguments.target),False,config,clip)
        cut_frames=shots.detect_hard_cuts(clip,config['shots'])['cuts'];windows=None;primary=arguments.speaker
    start,candidate=find_start(points,data,cut_frames,windows,primary,parse_intervals(arguments.others_speaking),config)
    # Preserve source-time registrations: initialize each currently visible person
    # from the full-video identity track, keep later explicit source anchors.
    fps=float(points['fps']);frame=round(start*fps)
    def rebase(anchors,person_points):
        values=[(n,t-start) for n,t in anchors if t>=start]
        index=int(person_points['target_index'][frame])
        ordered=track.sorted_person_indices(data['poses_all'],frame,float(data['width']),float(data['height']))
        if index in ordered: values.insert(0,(ordered.index(index)+1,0.))
        if not values:
            available=np.flatnonzero(np.asarray(person_points['target_index'])[frame:]>=0)
            if len(available):
                later=frame+int(available[0]);index=int(person_points['target_index'][later]);ordered=track.sorted_person_indices(data['poses_all'],later,float(data['width']),float(data['height']))
                if index in ordered: values.append((ordered.index(index)+1,later/fps-start))
        if not values: raise ValueError('--start auto: registered person never appears after selected start')
        return values
    if len(arguments.persons)==2:
        arguments.persons={name:rebase(anchors,_load_npz(full/f'person_{p}.npz')) for p,(name,anchors) in enumerate(arguments.persons.items())}
    else:
        arguments.target=[f'{n}@{t}' for n,t in rebase(_parse_targets(arguments.target),points)]
    arguments.auto_start={'actual_start':start,'sustained_state_start':candidate,'source_cache':str(full),'source_others_speaking':list(arguments.others_speaking),'source_speaker_map':list(arguments.speaker_map)}
    arguments.others_speaking=[f'{max(0.,a-start):g}-{b-start:g}' for a,b in parse_intervals(arguments.others_speaking) if b>start]
    if arguments.speaker_map:
        from .active_speaker import parse_maps
        arguments.speaker_map=[f'{max(0.,a-start):g}-{b-start:g}={name}' for a,b,name in parse_maps(arguments.speaker_map,list(arguments.persons)) if b>start]
    arguments.start=start
    return start
