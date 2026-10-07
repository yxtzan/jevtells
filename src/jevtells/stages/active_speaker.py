"""Audiovisual speaker evidence and explicit interval overrides, without voiceprints."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from .presence import presence_ratio
from .speakers import parse_intervals


def parse_maps(values, names):
    rows=[]
    for raw in values:
        interval,separator,name=raw.partition('=')
        if not separator or name not in names: raise ValueError('--speaker-map requires START-END=registered NAME')
        a,b=parse_intervals([interval])[0]
        if any(max(a,c)<min(b,d) for c,d,_ in rows): raise ValueError('overlapping --speaker-map intervals are ambiguous')
        rows.append((a,b,name))
    return sorted(rows)


def classify(window, tracks, opening, audio_t, envelope, fps, settings, has_voice=True):
    times=np.arange(len(opening))/fps;mask=(times>=window['t0'])&(times<window['t1'])
    audio=np.interp(times,audio_t,envelope)
    audio=uniform_filter1d(audio,size=max(1,round(settings['audio_smooth_seconds']*fps)))
    evidence={};ranked=[];observable=[]
    for p,(name,points) in enumerate(tracks.items()):
        indices=np.asarray(points['target_index']);mouth=np.full(len(opening),np.nan)
        for t,index in enumerate(indices):
            if index>=0: mouth[t]=opening[t,index]
        valid=mask&np.isfinite(mouth)
        motion=float(np.std(mouth[valid])) if valid.any() else None
        observed=float(valid.sum()/max(1,mask.sum()))
        corr=None
        if valid.sum()>=settings['min_samples'] and observed>=settings['mouth_min_observed_ratio']:
            observable.append(name)
            if np.std(audio[valid])>1e-9 and np.std(mouth[valid])>1e-9:
                corr=float(np.corrcoef(mouth[valid],audio[valid])[0,1])
                ranked.append((corr,name))
        evidence[name]=dict(correlation=corr,mouth_motion=motion,observed_ratio=observed,in_frame=bool((indices[mask]>=0).any()),presence_ratio=presence_ratio(points,window['t0'],window['t1']))
    ranked.sort(reverse=True);speaker=None;confidence=0.;reason='unknown'
    voiced=bool(has_voice and mask.any() and np.mean(audio[mask])>=settings['audio_min_rms'])
    if voiced and ranked:
        best,name=ranked[0];second=ranked[1][0] if len(ranked)>1 else 0.
        if best>=settings['correlation'] and best-second>=settings['correlation_margin'] and evidence[name]['mouth_motion']>=settings['mouth_motion']:
            speaker=name;confidence=max(0.,min(1.,best));reason='correlation'
    if voiced and speaker is None and len(observable)==1 and len(tracks)==2:
        name=observable[0]
        if evidence[name]['mouth_motion'] is not None and evidence[name]['mouth_motion']<settings.get('mouth_still_motion',settings['mouth_motion']):
            speaker=next(n for n in tracks if n!=name);reason='offscreen';confidence=1-min(1.,evidence[name]['mouth_motion']/settings.get('mouth_still_motion',settings['mouth_motion']))
    return dict(speaker=speaker,confidence=confidence,evidence=evidence,reason=reason)


def run(windows, tracks, opening, voice, config, out, maps=(), transcript=None):
    fps=float(next(iter(tracks.values()))['fps']);audio_t=np.asarray(voice['t']);envelope=10**(np.asarray(voice['rms_db'])/20)
    rows=[];marked=[]
    for window in windows:
        has_voice=window.get('kind')!='silence'
        if transcript is not None:
            has_voice=has_voice and any(max(float(s['t0']),window['t0'])<min(float(s['t1']),window['t1']) for s in transcript.get('segments',[]))
        result=classify(window,tracks,opening,audio_t,envelope,fps,config['two_person'],has_voice)
        midpoint=(window['t0']+window['t1'])/2
        override=next((name for a,b,name in maps if a<=midpoint<b),None)
        if override:
            result.update(speaker=override,confidence=1.,reason='manual')
        name=result['speaker'];ratio=result['evidence'].get(name,{}).get('presence_ratio',0.)
        row={**window,'speaker':name,'speaker_color':config['two_person']['colors'][list(tracks).index(name)] if name else config['render']['colors']['muted'],'speaker_unknown':name is None,'speaker_confidence':result['confidence'],'speaker_reason':result['reason'],'speaker_other':False,'target_offscreen':name is not None and ratio<.5,'target_presence_ratio':ratio,'two_person':True,'hold_previous_panel':False}
        marked.append(row);rows.append({'id':window['id'],'t0':window['t0'],'t1':window['t1'],**result})
    Path(out,'speakers.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return marked
