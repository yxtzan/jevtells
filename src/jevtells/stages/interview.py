"""Two-person orchestration isolated from the v0.6 single-target pipeline."""
from pathlib import Path
import json,time
import numpy as np
from . import prepare,pose,asr,voice,ocr,shots,segment,state,scene,judge,narrate,review,debug,mouth,active_speaker,two_person_track
from .. import render
from ..render.geometry import parse_blurs


def run(arguments,config):
    from ..cli import _clip_id,_run_actions,_finish_run
    started=time.perf_counter();times={};persons=arguments.persons
    maps=active_speaker.parse_maps(arguments.speaker_map,list(persons))
    input_path=Path(arguments.input);out=Path(arguments.output).parent if arguments.output else Path('work')/(_clip_id(input_path,arguments.start,arguments.duration)+'_two')
    out.mkdir(parents=True,exist_ok=True)
    if hasattr(arguments,"auto_start"):
        write(out,"run_meta.json",{"actual_start":arguments.start,"auto_start":arguments.auto_start})
    def step(name,function,*args,**kwargs):
        print(f'{name}: started',flush=True);tick=time.perf_counter();value=function(*args,**kwargs);times[name]=time.perf_counter()-tick;print(f'{name}: complete {times[name]:.1f}s',flush=True);return value
    force=arguments.force
    if arguments.until=='people':
        from .people import run as people_run
        clip,_,_=prepare.run(input_path,out,False,arguments.start,arguments.duration,config=config)
        data=pose.run(clip,out,False,config=config)
        print(people_run(clip,out,at=0.,detections=data,config=config));return
    if arguments.from_stage=='render':
        from ..cli import _load_npz
        windows=json.loads((out/'windows.json').read_text())
        previous=json.loads((out/'run_meta.json').read_text())['parameters']
        if previous.get('persons')!=json.loads(json.dumps(persons)) or previous.get('speaker_map',[])!=arguments.speaker_map:
            raise ValueError('person anchors or speaker maps changed; rerun analysis before --from render')
        render.run(out/'clip.mp4',out,windows,_load_npz(out/'keypoints.npz'),config=config,speaker=next(iter(persons)),lang=arguments.lang,title=arguments.title,layout=arguments.layout,blur=parse_blurs(arguments.blur),subtitles=arguments.subtitles=='on',force=False,debug_layout=arguments.debug_layout,reframe=arguments.reframe)
        return
    clip,audio,encoding=step('prepare',prepare.run,input_path,out,force,arguments.start,arguments.duration,config=config)
    if arguments.until=='prepare': return
    detections=step('pose',pose.run,clip,out,force,config=config)
    if arguments.until=='pose': return
    tracks=step('track',two_person_track.run,clip,detections,persons,out,config,force)
    if arguments.until=='track': return
    primary=next(iter(tracks));points=tracks[primary]
    audio_transcript=step('asr',asr.run,audio,out,None,force,model_name=config['models']['whisper'],destination_name='transcript_asr.json')
    detection=None
    if arguments.subtitle_source=='ocr' or (arguments.subtitle_source=='auto' and not arguments.srt and arguments.subtitles=='off'):
        from ..render.reframe import detect_subtitles
        detection=detect_subtitles(clip,config['render']['reframe'])
    source=ocr.resolve_source(arguments.subtitle_source,arguments.srt,arguments.subtitles,bool(detection and detection.get('bounds')))
    if source=='ocr': transcript=ocr.run(clip,out,audio_transcript['language'],config,force=force,detection=detection)
    elif source=='srt': transcript=asr._parse_srt(Path(arguments.srt));transcript['language']=audio_transcript['language']
    else: transcript=dict(audio_transcript)
    transcript['subtitle_source']=source;write(out,'transcript.json',transcript)
    if arguments.until=='asr': return
    voice_features=step('voice',voice.run,audio,out,force,config=config,transcript=audio_transcript)
    if arguments.until=='voice': return
    # Framing follows whichever registered person is visible in a single shot.
    framed={k:np.array(v,copy=True) for k,v in points.items()}
    for i in range(len(framed['pose'])):
        chosen=next((v for v in tracks.values() if v['target_index'][i]>=0),None)
        if chosen:
            for key in ('pose','hands','pose_present','hand_present','target_index'): framed[key][i]=chosen[key][i]
    shot_list=step('shots',shots.run,clip,out,True,points=framed,config=config,detections=detections)
    np.savez(out/'keypoints.npz',**framed)
    if arguments.until=='shots': return
    opening=step('mouth',mouth.read_mouths,clip,detections['poses_all'],out,config,force)
    windows=segment.run(transcript,shot_list,out,True,config)
    if maps:
        from .speakers import split_speaker_changes
        windows=split_speaker_changes(windows,[(a,b) for a,b,_ in maps],transcript,config)
    windows=active_speaker.run(windows,tracks,opening,voice_features,config,out,maps,audio_transcript)
    windows=segment.merge_short_windows(windows,config,boundaries=[t for a,b,_ in maps for t in (a,b)])
    windows=active_speaker.run(windows,tracks,opening,voice_features,config,out,maps,audio_transcript)
    write(out,'windows.json',windows);voice_features=voice.refresh_window_metrics(voice_features,windows,audio_transcript,out,config)
    if arguments.until=='segment': return
    actions={'events':[],'discarded':[],'occupied_hands':{}}
    for p,(name,person_points) in enumerate(tracks.items()):
        folder=out/f'person_{p}';folder.mkdir(exist_ok=True)
        # Per-person shot scale/presence is measured independently.
        person_shots=shots.run(clip,folder,True,points=person_points,config=config,detections=detections)
        # Action cut indices use the common framing clock of the renderer.
        for row in person_shots:
            row['index']=next((s['index'] for s in shot_list if s['t0']<=row['t0']<s['t1']),row['index'])
        write(folder,'shots.json',person_shots)
        result=_run_actions(person_points,windows,folder,True,config,clip)
        for field in ('events','discarded'):
            for event in result.get(field,[]):
                display_shot=next((s['index'] for s in shot_list if s['t0']<=(event['t0']+event['t1'])/2<s['t1']),None)
                actions[field].append({**event,'shot_index':display_shot,'id':f'P{p}_{event.get("id",len(actions[field]))}','person':name})
        actions['occupied_hands'][name]=result.get('occupied_hands',{})
    write(out,'actions.json',actions)
    if arguments.until=='actions': return
    scene_result=step('scene',scene.run,clip,out,force,config=config,explicit=arguments.scene)
    previous_states=json.loads((out/'states.json').read_text()) if (out/'states.json').exists() else None
    states=step('state',state.run,windows,voice_features,scene_result['scene'],primary,out,True,tracks,actions,config,audio_transcript)
    debug.run(clip,out,windows,framed,encoding['encoder'],force,detections=detections,actions=actions,config=config)
    arguments.speaker=primary
    def finish(): _finish_run(arguments,out,clip,encoding,transcript,persons[primary],actions,started,times,windows)
    if arguments.until in {'scene','state','debug'}: finish();return
    judgments=step('judge',judge.run,states,out,force or states!=previous_states,config=config)
    if arguments.until=='judge': finish();return
    narration=step('narrate',narrate.run,states,judgments,out,force or states!=previous_states,config=config,lang=arguments.lang)
    review.write(states,judgments,narration,out,lang=arguments.lang)
    if arguments.until=='render':
        step('render',render.run,clip,out,windows,framed,config=config,speaker=primary,lang=arguments.lang,title=arguments.title,layout=arguments.layout,blur=parse_blurs(arguments.blur),subtitles=arguments.subtitles=='on',force=force,debug_layout=arguments.debug_layout,reframe=arguments.reframe)
    finish()


def write(out,name,value):
    Path(out,name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
