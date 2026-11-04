"""Opt-in sustained single-target start, ported from M7 without dual mode."""
from pathlib import Path
import numpy as np
from . import prepare, track, shots
from .framing import person_mask


def find_start(points, detections, cuts, intervals, config):
    fps=float(points['fps']);poses=detections['poses_all'];settings=config['shots']
    required=max(1,round(config['auto_start']['sustained_seconds']*fps))
    count=0;candidate=None
    for i in range(len(poses)):
        people=person_mask(poses[i],{**settings,'visibility':config['detection']['pose_visibility_threshold']})
        t=i/fps
        eligible=points['target_index'][i]>=0 and int(people.sum())==1 and not any(a<=t<b for a,b in intervals)
        count=count+1 if eligible else 0
        if count>=required:
            candidate=(i-required+1)/fps;break
    if candidate is None:
        raise ValueError('--start auto: no sustained solo target outside --others-speaking; use --start SECONDS')
    # Never align across other-speaker intervals or a non-solo shot.
    boundaries=[0.,*[c/fps for c in cuts]]
    aligned=max((t for t in boundaries if t<=candidate+1e-6),default=candidate)
    start_frame=round(aligned*fps); candidate_frame=round(candidate*fps)
    for i in range(start_frame,candidate_frame):
        t=i/fps
        people=person_mask(poses[i],{**settings,'visibility':config['detection']['pose_visibility_threshold']})
        if points['target_index'][i]<0 or people.sum()!=1 or any(a<=t<b for a,b in intervals):
            aligned=candidate;break
    return float(aligned),float(candidate)


def rebase(anchors, points, data, start, duration=None):
    fps=float(points['fps']);frame=round(start*fps)
    end=start+duration if duration is not None else float('inf')
    values=[(n,t-start) for n,t in anchors if start<=t<end]
    if frame>=len(points['target_index']):
        raise ValueError('Selected start is outside video')
    available=np.flatnonzero(np.asarray(points['target_index'])[frame:]>=0)
    if len(available):
        later=frame+int(available[0]);index=int(points['target_index'][later])
        ordered=track.sorted_person_indices(data['poses_all'],later,float(data['width']),float(data['height']))
        if index in ordered and later/fps<end:
            values.insert(0,(ordered.index(index)+1,later/fps-start))
    if not values:
        raise ValueError('Registered target never appears in the selected clip')
    return sorted(set(values), key=lambda pair:pair[1])


def source_track(arguments, config, anchors, directory):
    from ..cli import _run_pose
    clip,_,_=prepare.run(Path(arguments.input),directory,False,0.,None,config)
    data=_run_pose(clip,directory,False,config)
    points=track.run(data,directory,anchors,False,config,clip)
    return data,points,clip


def resolve(arguments, config):
    from ..cli import _parse_targets, _clip_id
    from .speakers import parse_intervals
    from .pick import resolve as resolve_pick
    full=Path('work')/(Path(arguments.input).stem+'_autostart')
    canonical=Path('work')/_clip_id(Path(arguments.input),0.,None)
    anchors,picked=resolve_pick(arguments.target, full, canonical,Path(arguments.input))
    if arguments.target_desc:
        # Description is resolved against the full source before rebasing.
        from .target_desc import select
        clip,_,_=prepare.run(Path(arguments.input),full,False,0.,None,config)
        from ..cli import _run_pose
        data=_run_pose(clip,full,False,config)
        anchors=[select(clip,data,arguments.target_desc,arguments.target_at,full/'target_desc',config)]
        arguments.target_desc=None
    data,points,clip=source_track(arguments,config,anchors,full)
    if not arguments.target and not picked:
        visible=track.sorted_person_indices(data['poses_all'],0,float(data['width']),float(data['height']))
        if len(visible)!=1:
            raise ValueError('--start auto requires pick or --target for a multi-person opening')
    intervals=parse_intervals(arguments.others_speaking)
    cuts=shots.detect_hard_cuts(clip,config['shots'])['cuts']
    start,candidate=find_start(points,data,cuts,intervals,config)
    arguments.target=[f'{n}@{t:.9f}' for n,t in rebase(anchors,points,data,start,arguments.duration)]
    arguments.auto_start={'actual_start':start,'sustained_state_start':candidate,'source_cache':str(full),
                          'source_anchors':anchors,'source_others_speaking':list(arguments.others_speaking)}
    arguments.others_speaking=[f'{max(0.,a-start):.9f}-{b-start:.9f}' for a,b in intervals if b>start]
    arguments.start=start
    if picked:
        print(f'使用 pick 选定的目标：{anchors} ({picked})')
