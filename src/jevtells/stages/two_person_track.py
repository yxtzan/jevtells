"""Fixed face/clothing references, conservative exclusive two-person tracking."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import appearance, face_identity, track, shots
from ..utils.geometry import match_hands_to_pose


def exclusive_assignment(scores, eligible):
    """Dummies allow absence; incompatible candidates never force a match."""
    count, candidates=scores.shape
    matrix=np.concatenate([np.where(eligible,scores,-1e6),np.zeros((count,count))],axis=1)
    rows,cols=linear_sum_assignment(-matrix)
    result=np.full(count,-1,int)
    for r,c in zip(rows,cols):
        if c<candidates and eligible[r,c] and matrix[r,c]>0: result[r]=c
    return result


def person_points(data, indices, config):
    poses=data['poses_all'];hands=data['hands_all'];n=len(poses)
    pose=np.full((n,33,4),np.nan,np.float32);selected_hands=np.full((n,2,21,3),np.nan,np.float32)
    w,h=float(data['width']),float(data['height'])
    for t,p in enumerate(indices):
        if p<0: continue
        pose[t]=poses[t,p]
        shoulder=track._shoulder_width(pose[t],w,h)
        assignments=match_hands_to_pose(hands[t,:,0,:],pose[t],(w,h),max_distance=config['detection']['hand_match_max_shoulder_width']*shoulder)
        for hand,side in enumerate(assignments):
            if side>=0: selected_hands[t,side]=hands[t,hand]
    return dict(fps=data['fps'],width=data['width'],height=data['height'],n_frames=n,t=data['t'],pose=pose,hands=selected_hands,target_index=indices,pose_present=indices>=0,hand_present=np.isfinite(selected_hands[...,:2]).all(axis=3).any(axis=2))


def assign_tracks(poses, embeddings, clothing, persons, cuts, fps, width, height, config):
    names=list(persons);rules=track._config(config);settings=config['two_person'];n,k=poses.shape[:2]
    face_refs=[face_identity.anchor_references(poses,embeddings,persons[name],cuts,fps,width,height,config) for name in names]
    cloth_indices=[];cloth_meta=[];cloth_refs=[]
    for name in names:
        indices,meta=appearance.track_appearance(poses,clothing,persons[name],cuts,fps,width,height,config)
        cloth_indices.append(indices);cloth_meta.append(meta)
        refs=[]
        for a in meta['references']:
            start=a['frame'];stop=round(a['end']*fps)
            samples=[clothing[t,indices[t]] for t in range(start,stop) if indices[t]>=0 and np.isfinite(clothing[t,indices[t]]).all()]
            if samples:
                ref=np.median(samples,axis=0);ref/=ref.sum(axis=1,keepdims=True);refs.append(ref)
        cloth_refs.append(refs)
    indices=np.full((n,len(names)),-1,int);face_scores=np.full((n,len(names),k),-1.);cloth_scores=np.zeros((n,len(names),k));seconds=np.zeros((n,len(names)))
    anchors={}
    for p,name in enumerate(names):
        for number,t in persons[name]: anchors[round(t*fps),p]=track.select_anchor(poses,round(t*fps),number,width,height,rules.visibility_threshold)
    previous=[None]*len(names)
    for t in range(n):
        if t==0 or t in cuts: previous=[None]*len(names)
        scores=np.zeros((len(names),k));eligible=np.zeros((len(names),k),bool)
        for p in range(len(names)):
            for c in range(k):
                face_scores[t,p,c]=face_identity.cosine(embeddings[t,c],face_refs[p])
                cloth_scores[t,p,c]=appearance.similarity(clothing[t,c],cloth_refs[p])
            ranked=sorted(face_scores[t,p],reverse=True);seconds[t,p]=ranked[1] if k>1 else -1
            if (t,p) in anchors:
                c=anchors[t,p]
                if c>=0: scores[p,c]=10;eligible[p,c]=True
                continue
            has_face=np.isfinite(embeddings[t]).all(axis=1)
            for c in range(k):
                face=face_scores[t,p,c];cloth=cloth_scores[t,p,c]
                # Available but mismatched faces cannot be overridden by shirt colour.
                if has_face[c] and face_refs[p]:
                    others=np.delete(face_scores[t,p],c)
                    margin=face-max(others,default=-1)
                    if face>=settings['face_cosine'] and margin>=settings['face_margin']:
                        eligible[p,c]=True;scores[p,c]=2+face+.01*cloth
                elif cloth_indices[p][t]==c:
                    eligible[p,c]=True;scores[p,c]=cloth
            # No identity learning from predictions: all references remain anchors.
        indices[t]=exclusive_assignment(scores,eligible)
    metadata={}
    edges=sorted({0,n,*cuts})
    for p,name in enumerate(names):
        rows=[]
        for a,b in zip(edges,edges[1:]):
            selected=indices[a:b,p];valid=selected>=0
            fs=[float(face_scores[t,p,c]) for t,c in zip(range(a,b),selected) if c>=0]
            cs=[float(cloth_scores[t,p,c]) for t,c in zip(range(a,b),selected) if c>=0]
            rows.append(dict(t0=a/fps,t1=b/fps,result='recognized' if valid.any() else 'offscreen',presence_ratio=float(valid.mean()),face_similarity=float(np.median(fs)) if fs else None,clothing_similarity=float(np.median(cs)) if cs else None,second_similarity=float(np.median(seconds[a:b,p])),anchor_frames=[t for t,who in anchors if who==p and a<=t<b]))
        metadata[name]={'face_reference_samples':len(face_refs[p]),'clothing_references':cloth_meta[p]['references'],'shots':rows}
    return indices,metadata


def run(clip, data, persons, out, config, force=False):
    out=Path(out);signature={'persons':persons,'config':config}
    meta_path=out/'track_meta.json'
    if meta_path.exists() and not force:
        meta=json.loads(meta_path.read_text())
        if meta.get('signature')==json.loads(json.dumps(signature)):
            result={}
            for p,name in enumerate(persons):
                with np.load(out/f'person_{p}.npz',allow_pickle=False) as z: result[name]={k:z[k] for k in z.files}
            return result
    poses=data['poses_all'];features=face_identity.read_features(clip,poses,out,config,force)
    cuts=shots.detect_hard_cuts(clip,config['shots'])['cuts']
    clothing=appearance.read_appearances(clip,poses,track._config(config).visibility_threshold)
    indices,meta=assign_tracks(poses,features['embeddings'],clothing,persons,cuts,float(data['fps']),float(data['width']),float(data['height']),config)
    result={}
    for p,name in enumerate(persons):
        result[name]=person_points(data,indices[:,p],config);np.savez(out/f'person_{p}.npz',**result[name])
    np.savez(out/'keypoints.npz',**result[next(iter(persons))])
    meta_path.write_text(json.dumps({'signature':signature,'persons':meta,'cuts':cuts,'identity_version':2},ensure_ascii=False,indent=2))
    return result
