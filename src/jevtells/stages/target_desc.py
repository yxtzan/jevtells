"""Two-step text selection; no speculative Box when presence is uncertain."""
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
from ..clients.openrouter import OpenRouterClient
from . import jev_images, track


def match_box(answer, poses, width, height, settings, visibility):
    box = answer.get('box',{})
    edges = [float(box[k]) for k in ('left','top','right','bottom')]
    if not all(math.isfinite(v) and 0<=v<=1 for v in edges) or edges[0]>=edges[2] or edges[1]>=edges[3]:
        raise ValueError('Invalid Jev Box')
    pixels = np.array(edges)*[width,height,width,height]
    ranked = sorted([(track.bbox_iou(pixels,track.person_box(p,width,height,visibility)),i) for i,p in enumerate(poses)],reverse=True)
    if not ranked or ranked[0][0]<settings['min_iou'] or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<settings['iou_margin']):
        raise ValueError('Description did not overlap a unique detected person sufficiently; use pick or --target')
    return ranked[0][1], ranked


def select(clip, detections, description, at, out, config, client=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fps=float(detections['fps']); index=round(at*fps)
    if not description.strip() or not 0<=index<len(detections['poses_all']):
        raise ValueError('Description and anchor time must be valid')
    frame=jev_images.read_frame(clip,index); original_h,original_w=frame.shape[:2]
    image=jev_images.compliant(frame,config['jev_images']);path=out/'target_desc.jpg';cv2.imwrite(str(path),image)
    # Full frames fit the limits without padding in ordinary 16:9 footage.
    # Box coordinates must refer to the same resized/padded image as the poses.
    scaled_poses=detections['poses_all'][index].copy()
    h,w=image.shape[:2]; scale=min(1.,config['jev_images']['preferred_side']/max(original_w,original_h),config['jev_images']['max_side']/max(original_w,original_h))
    rw,rh=max(1,round(original_w*scale)),max(1,round(original_h*scale))
    scaled_poses[:,:,0]=(scaled_poses[:,:,0]*rw+(w-rw)//2)/w
    scaled_poses[:,:,1]=(scaled_poses[:,:,1]*rh+(h-rh)//2)/h
    reference,item=jev_images.asset(path);assets={reference:item};state={'frame':reference,'description':description}
    client=client or OpenRouterClient(attempts=1,audit_dir=out)
    settings=config['target_desc'];started=time.perf_counter()
    presence=client.decide(state,{'present':{'type':'noul','instructions':'There is exactly one visible person matching `description` in `frame`.','criteria':{'true':'one matching person is clearly present','false':'absent, ambiguous or not observable'}}},settings['model'],assets=assets)
    (out/'presence_raw.json').write_text(json.dumps(presence,ensure_ascii=False,indent=2))
    probability=float(presence['raw']['answers']['present']['noul'])
    if not math.isfinite(probability) or probability<settings['presence_probability']:
        raise ValueError('No unambiguous person matching the description; use pick or --target')
    result=client.decide(state,{'person':{'type':'box','instructions':'The person matching `description` in `frame`.'}},settings['model'],assets=assets)
    (out/'box_raw.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    selected,ranked=match_box(result['raw']['answers']['person'],scaled_poses,w,h,settings,config['detection']['pose_visibility_threshold'])
    ordered=track.sorted_person_indices(detections['poses_all'],index,float(detections['width']),float(detections['height']),config['detection']['pose_visibility_threshold'])
    number=ordered.index(selected)+1
    receipt={'description':description,'at':index/fps,'person_number':number,'detector_index':selected,'presence':probability,'iou_scores':ranked,'elapsed_s':time.perf_counter()-started,'cost':sum(r['cost'] for r in (presence,result))}
    (out/'selection.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return number,index/fps
