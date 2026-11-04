"""Prepare M8 image review, optionally run live comparisons within a ledger.

Default is offline: existing 1.13 judgments are explicitly labeled historical,
missing experiments remain absent. --live requests all three groups and speech
checks; --render additionally creates the image-based Tim release candidate.
No ground-truth answer is ever sent to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jevtells.config import load_config
from jevtells.clients.openrouter import OpenRouterClient
from jevtells.stages import jev_images,judge,narrate,target_desc
from jevtells.cli import _load_npz
from jevtells import render
from jevtells.render.geometry import parse_blurs


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def describe(judgment):
    if judgment is None:
        return 'skipped'
    scores=', '.join(f'{key}={value.get("value") if value else None}' for key,value in judgment.get('scores',{}).items())
    choices=', '.join(f'{key}={value.get("label")} ({value.get("confidence")})' for key in ('intent','emotion') if (value:=judgment.get(key)))
    return f'{scores}; {choices}'


def summarize_comparison(groups, threshold):
    """Use the same completed windows; score answers have no confidence field."""
    from itertools import combinations
    from statistics import mean
    counts={}
    for group,rows in groups.items():
        valid=[r for r in rows.values() if r['status']=='completed']
        confidence={key:[r['judgment'][key]['confidence'] for r in valid
                         if r['judgment'].get(key) and r['judgment'][key].get('confidence') is not None]
                    for key in ('intent','emotion')}
        combined=confidence['intent']+confidence['emotion']
        counts[group]={'completed':len(valid),
                      'average_intent_confidence':mean(confidence['intent']) if confidence['intent'] else None,
                      'average_emotion_confidence':mean(confidence['emotion']) if confidence['emotion'] else None,
                      'average_choice_confidence':mean(combined) if combined else None,
                      'confidence_values':len(combined),
                      'not_obvious_windows':sum(any((r['judgment'].get(key) or {}).get('confidence') is None or r['judgment'][key]['confidence']<threshold for key in ('intent','emotion')) for r in valid) if valid else None,
                      'cost':sum(r['cost'] for r in valid) if valid and all(r.get('cost') is not None for r in valid) else None,
                      'elapsed_s':sum(r['elapsed_s'] for r in valid) if valid else None}
    def compare(names):
        identifiers=set.intersection(*(set(groups[name]) for name in names))
        comparable=[identifier for identifier in identifiers if all(groups[name][identifier]['status']=='completed' for name in names)]
        different={key:[identifier for identifier in sorted(comparable) if len({(groups[name][identifier]['judgment'].get(key) or {}).get('label') for name in names})>1] for key in ('intent','emotion')}
        return {'comparable_windows':len(comparable),'intent_different':len(different['intent']),
                'emotion_different':len(different['emotion']),'either_different':len(set(different['intent'])|set(different['emotion'])),
                'window_ids':different}
    differences={'all_three':compare(list(groups)),
                 'pairwise':{' vs '.join(pair):compare(pair) for pair in combinations(groups,2)}}
    return counts,differences


def evaluate(directory, name, out, config, live=False):
    destination=out/name;destination.mkdir(parents=True,exist_ok=True)
    windows=read(directory/'windows.json');states=read(directory/'states.json')
    # Five-field source is mandatory; image paths are added to a copy only.
    states={k:{key:value for key,value in row.items() if key!='frames'} for k,row in states.items()}
    points=_load_npz(directory/'keypoints.npz');actions=read(directory/'actions.json')
    frames=jev_images.extract(directory/'clip.mp4',destination,windows,points,actions,config['jev_images'])
    faces=jev_images.extract(directory/'clip.mp4',destination,windows,points,actions,config['jev_images'],True)
    historical=read(directory/'judgments.json');rows=[];groups={};speaking=[]
    client=OpenRouterClient(attempts=1,audit_dir=destination/'raw')
    base_questions=judge.load_questions()
    group_specs=(('1.13-text','typesafe/jev-1.13',False),('1.14-text','typesafe/jev-1.14',False),('1.14-images','typesafe/jev-1.14',True))
    for group,model,images in group_specs:
        groups[group]={}
        for window in windows:
            identifier=window['id'];path=destination/group/f'{identifier}.json'
            if path.exists():
                result=read(path)
            elif window.get('speaker_other') or window.get('target_offscreen'):
                result={'status':'skipped','judgment':None,'reason':'speaker_other' if window.get('speaker_other') else 'target_offscreen'}
            elif live:
                state=states[identifier];questions=judge._actions_questions(state,base_questions)
                kwargs={}
                if images:
                    state,assets=jev_images.bind({**state,'frames':frames[identifier]})
                    kwargs['assets']=assets
                    questions={key:{**q,'instructions':str(q.get('instructions',q.get('question','')))+' Also consider expressions and posture in `frames`.'} for key,q in questions.items()}
                started=time.perf_counter();response=client.decide(state,questions,model,**kwargs)
                result={'status':'completed','raw':response['raw'],'judgment':judge.normalise_judgment(response['raw'],questions,judge._action_ids(states[identifier],identifier,directory)),
                        'cost':response['cost'],'elapsed_s':time.perf_counter()-started,'model':model}
            else:
                result={'status':'not_run','reason':'Live API required; last smoke returned HTTP 401 API key expired'}
            if result['status']!='not_run':
                write(path,result)
            groups[group][identifier]=result
    for window in windows:
        identifier=window['id'];path=destination/'speaking'/f'{identifier}.json'
        if path.exists():
            result=read(path)
        elif live and faces[identifier]:
            state,assets=jev_images.bind({'subtitle':states[identifier]['subtitle']['current'],'frames':faces[identifier]})
            started=time.perf_counter()
            response=client.decide(state,{'speaking':{'type':'noul','instructions':'The current subtitle was spoken by the person shown in the face `frames`.','criteria':{'true':'The pictured person is speaking this subtitle','false':'Someone else is speaking'}}},config['jev_images']['model'],assets=assets)
            probability=float(response['raw']['answers']['speaking']['noul'])
            high=config['speaking_images']['positive_probability'];low=config['speaking_images']['negative_probability']
            result={'status':'completed','probability':probability,'prediction':True if probability>=high else False if probability<=low else None,
                    'elapsed_s':time.perf_counter()-started,'cost':response['cost'],'raw':response['raw']}
            write(path,result)
        else:
            result={'status':'not_run' if not live else 'unobservable','prediction':None}
        # Reference comes from the documented intervals in the source run.
        intervals=read(directory/'run_meta.json')['parameters']['others_speaking']
        pairs=[tuple(map(float,item.split('-'))) for item in intervals]
        overlap=sum(max(0,min(window['t1'],b)-max(window['t0'],a)) for a,b in pairs)
        reference=not (overlap/(window['t1']-window['t0'])>=config['other_speaker_overlap'])
        speaking.append({'window':identifier,**result,'reference_is_target':reference,'reference_other_seconds':overlap})
    observed=[r for r in speaking if r['status']=='completed']
    metric={'reference':'docs/samples.md intervals copied from v0.6.0 run_meta; majority overlap per window',
            'windows':len(speaking),'completed':len(observed),'correct':sum(r['prediction']==r['reference_is_target'] for r in observed),
            'accuracy_all_reference_windows':sum(r['prediction']==r['reference_is_target'] for r in observed)/len(speaking) if observed else None,
            'coverage':sum(r.get('prediction') is not None for r in speaking)/len(speaking) if observed else None,
            'accuracy_completed_including_abstentions':sum(r['prediction']==r['reference_is_target'] for r in observed)/len(observed) if observed else None,
            'unknown':sum(r.get('prediction') is None for r in speaking),'rows':speaking}
    write(destination/'speaking_eval.json',metric)
    lines=[f'## {name}','','Historical 1.13 is a v0.6.0 cache reference, not a new M8 request. `not_run` means no result exists.','','| Window / subtitle | Target frames | Historical 1.13 | M8 1.13 text | M8 1.14 text | M8 1.14 images |','|---|---|---|---|---|---|']
    for window in windows:
        identifier=window['id'];pictures=' '.join(f'<img src="{Path(p).relative_to(out.resolve())}" width="150">' for p in frames[identifier]) or 'Target offscreen'
        current=[]
        for group,_,_ in group_specs:
            r=groups[group][identifier]
            current.append(describe(r.get('judgment'))+f"; {r.get('elapsed_s')}s; ${r.get('cost')}" if r['status']=='completed' else r['status'])
        subtitle=window.get('subtitle','').replace('|','\\|').replace('\n',' ')
        lines.append(f'| {identifier} {window["t0"]:.3f}–{window["t1"]:.3f}s<br>{subtitle} | {pictures} | {describe(historical.get(identifier))} | '+ ' | '.join(current)+' |')
    counts,differences=summarize_comparison(groups,config['judgment']['min_confidence'])
    write(destination/'comparison.json',{'windows':windows,'groups':groups,'summary':counts,'differences':differences})
    return '\n'.join(lines),groups,frames


def evaluate_descriptions(directory,name,out,config,live):
    # Frame times and reference ordinals are evaluation inputs, never rules
    # inside target_desc.select. Each video has three multi-person frames.
    times=(1.5,2.,3.) if name=='tim' else (1.,3.,6.)
    examples=(('白T恤、戴眼镜的男士',1),('深色上衣、拿着笔记本的男士',2)) if name=='tim' else (('坐在中间、灰白头发、蓝色领带的男士',2),('坐在画面左侧、穿黑色西装的男士',1))
    data=_load_npz(directory/'detections.npz');rows=[]
    for t in times:
        for description,expected in examples:
            identifier=f'{name}_{t:g}_{expected}';destination=out/'descriptions'/identifier
            receipt=destination/'evaluation.json'
            if live and receipt.exists():
                rows.append(read(receipt))
                continue
            if live:
                started=time.perf_counter()
                try:
                    number,at=target_desc.select(directory/'clip.mp4',data,description,t,destination,config)
                    row={'selected':number,'correct':number==expected,'status':'completed','reason':None}
                except ValueError as error:
                    # Conservative rejection is an experimental result, not an
                    # API failure. Preserve it and evaluate the other examples.
                    row={'selected':None,'correct':None,'status':'rejected','reason':str(error)}
                responses=[read(path) for path in (destination/'presence_raw.json',destination/'box_raw.json') if path.exists()]
                row.update(elapsed_s=time.perf_counter()-started,
                           cost=sum(r['cost'] for r in responses) if all(r.get('cost') is not None for r in responses) else None)
                if (destination/'presence_raw.json').exists():
                    row['presence_probability']=read(destination/'presence_raw.json')['raw']['answers']['present']['noul']
                if (destination/'selection.json').exists():
                    row['selection']=read(destination/'selection.json')
            else:
                row={'selected':None,'correct':None,'status':'not_run: expired API key'}
            record={'video':name,'at':t,'description':description,'expected_person':expected,**row}
            rows.append(record)
            if live:
                write(receipt,record)
    return rows


def render_tim(directory,groups,frames,out,config):
    results=groups['1.14-images']
    if any(r['status'] not in ('completed','skipped') for r in results.values()):
        raise ValueError('Image group incomplete; refuse to render historical judgments as new results')
    destination=out/'tim_image_render';shutil.copytree(directory,destination,dirs_exist_ok=True)
    states=read(directory/'states.json');states={key:{**row,'frames':frames[key]} for key,row in states.items()}
    judgments={key:r.get('judgment') for key,r in results.items()}
    write(destination/'states.json',states);write(destination/'judgments.json',judgments)
    for path in (destination/'raw').glob('jev_W*.json'):
        path.unlink()
    for identifier,result in results.items():
        if result['status']=='completed':
            write(destination/'raw'/f'jev_{identifier}.json',result['raw'])
    write(destination/'judge_meta.json',{'model':config['jev_images']['model'],'calls':sum(r['status']=='completed' for r in results.values()),
                                       'cost':sum(r.get('cost',0) for r in results.values())})
    narrate.run(states,judgments,destination,True,config,'zh')
    params=read(directory/'run_meta.json')['parameters'];windows=read(directory/'windows.json')
    image_config=json.loads(json.dumps(config));image_config['jev']['model']=config['jev_images']['model']
    render.run(destination/'clip.mp4',destination,windows,_load_npz(destination/'keypoints.npz'),config=image_config,speaker=params['speaker'],title=params['title'],layout='v',blur=parse_blurs(params['blur']),subtitles=False,source_size=(1280,720),force=True)
    publish=Path('work/_publish/tim_jevimg_v.mp4');publish.parent.mkdir(exist_ok=True)
    shutil.copy2(destination/'output_v.mp4',publish)
    windows_to_review=[w for w in windows if results[w['id']]['status']=='completed'][:3]
    for window in windows_to_review:
        index=round((window['t0']+window['t1'])/2*float(config['render']['fps']))
        import cv2
        for label,path in [('before',directory/'output_v.mp4'),('images',publish)]:
            cv2.imwrite(str(destination/f'{window["id"]}_{label}.jpg'),jev_images.read_frame(path,index))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tim',type=Path,default=Path('work/M8/default_tim'))
    parser.add_argument('--jensen',type=Path,default=Path('work/M8/default_jensen'))
    parser.add_argument('--output',type=Path,default=Path('work/M8'))
    parser.add_argument('--ledger',type=Path,help='shared ledger for this API run, including smoke requests')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args();config=load_config();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.render and not args.live:
        raise ValueError('--render requires --live and a complete screenshot comparison')
    if args.live:
        os.environ['JEVTELLS_API_LEDGER']=str((args.ledger or out/'api_ledger.jsonl').resolve())
        os.environ['JEVTELLS_API_LIMIT']='0.30'
    content=['# M8 Jev images review','','Historical results are explicitly separated from new API responses. Speaker-other/offscreen skips follow the default single-person policy. Each completed cell reports actual API results, latency and cost.','']
    descriptions=[];tim_results=None
    for name,directory in (('tim',args.tim),('jensen',args.jensen)):
        text,groups,frames=evaluate(directory,name,out,config,args.live);content.extend((text,''))
        descriptions.extend(evaluate_descriptions(directory,name,out,config,args.live))
        if name=='tim':tim_results=(groups,frames)
    (out/'jev_images_review.md').write_text('\n'.join(content))
    write(out/'description_eval.json',descriptions)
    if args.render:
        render_tim(args.tim,*tim_results,out,config)
    print(out/'jev_images_review.md')

if __name__=='__main__':main()
