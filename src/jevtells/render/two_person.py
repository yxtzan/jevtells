"""Two independent label tracks sharing the same video and analysis panels."""
import copy
import numpy as np
from .renderer import Composer
from .body_zones import body_zones
from .layout_audit import intersects
from ..stages.shots import shoulder_ratios


class PersonComposer(Composer):
    def __init__(self,*args,other_points=None,reserved=None,**kwargs):
        self.other_points=other_points;self.reserved=reserved or {}
        super().__init__(*args,**kwargs)

    def forbidden(self,seconds):
        result=super().forbidden(seconds)
        index=min(len(self.points['pose'])-1,max(0,round(seconds*float(self.points['fps']))))
        if self.other_points is not None:
            zones=body_zones(np.asarray(self.other_points['pose'])[index],self.layout.source,self.visibility)
            if zones.get('torso'):
                x0,y0,x1,y1=zones['torso'];result.append({'name':'other_core','rect':list(self.layout.source_rect((x0,y0,x1-x0,y1-y0)))})
        shot=next((s for s in self.shots if s['t0']<=seconds<s['t1']),self.shots[-1])
        for rect in self.reserved.get(shot['index'],[]): result.append({'name':'other_label','rect':rect})
        return result


class TwoPersonComposer(Composer):
    def __init__(self,*args,person_tracks,**kwargs):
        super().__init__(*args,**kwargs)
        self.person_composers=[];reserved={}
        names=list(person_tracks)
        for p,name in enumerate(names):
            settings=copy.deepcopy(self.settings);settings['colors']['lime']=kwargs['config']['two_person']['colors'][p]
            track={**person_tracks[name],'poses_all':self.all_poses}
            events=[e for e in self.events if e.get('person')==name]
            person_shots=[]
            ratios=shoulder_ratios(track,*self.layout.source,self.visibility)
            fps=float(track['fps'])
            for shot in self.shots:
                a,b=round(shot['t0']*fps),round(shot['t1']*fps)
                valid=np.isfinite(ratios[a:b]);median=float(np.nanmedian(ratios[a:b])) if valid.any() else 0.
                person_shots.append({**shot,'label':'target' if np.any(track['target_index'][a:b]>=0) else 'other','far':median<kwargs['config']['shots']['far_shoulder_ratio']})
            child=PersonComposer(settings,self.base_layout,self.windows,track,events,self.judgments,self.panels.narration,self.transcript,title=self.panels.title,sources=self.panels.sources,lang=kwargs['lang'],blur=self.blur,subtitles=self.subtitles,config=kwargs['config'],shots=person_shots,reframe_plan=self.reframe_plan,other_points=person_tracks[names[1-p]],reserved=reserved)
            self.person_composers.append((name,child))
            for shot in child.shots:
                place=child.placements[shot['index']]
                if not events or shot['label']=='other' or shot['far']: continue
                for slot,xy in place['positions'].items():
                    if place['sample_counts'][slot]:
                        w,h=place['sizes'][slot];reserved.setdefault(shot['index'],[]).append([xy[0],xy[1],xy[0]+w,xy[1]+h])
        self.placements={s['index']:{'shot_index':s['index'],'t0':s['t0'],'t1':s['t1'],'persons':{name:child.placements[s['index']] for name,child in self.person_composers}} for s in self.shots}

    def _labels(self,image,seconds,point_index,lost,shot):
        for name,child in self.person_composers:
            child.layout=self.layout
            child.audit={'labels':[]}
            absent=int(child.points['target_index'][point_index])<0
            child._labels(image,seconds,point_index,absent,'target')
            for field in ('labels','expected_labels'):
                for label in child.audit.get(field,[]):
                    core=body_zones(np.asarray(child.points['pose'])[point_index],self.layout.source,self.visibility)
                    mid=self.layout.source_point(core['midline'],0)[0] if 'midline' in core else None
                    self.audit.setdefault(field,[]).append({**label,'slot':name+':'+label['slot'],'person':name,'midline':mid})
            for label in child.audit.get('labels',[]):
                # The auditor checks each label against other people's cores.
                label['person']=name
            for zone in child.forbidden(seconds):
                if zone['name']=='other_core':
                    self.audit.setdefault('person_forbidden',{})[name]=zone['rect']
