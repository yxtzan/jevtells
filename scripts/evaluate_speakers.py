"""Evaluate named audiovisual windows against source-time interval annotations."""
from pathlib import Path
import argparse,json
from collections import Counter


def evaluate(rows, intervals, default, start=0.):
    confusion=Counter();named=correct=unknown=mixed=0;details=[]
    for row in rows:
        a,b=row['t0']+start,row['t1']+start
        overlap={default:b-a}
        for c,d,name in intervals:
            seconds=max(0.,min(b,d)-max(a,c))
            overlap[default]-=seconds;overlap[name]=overlap.get(name,0)+seconds
        truth=max(overlap,key=overlap.get);ambiguous=max(overlap.values())<(b-a)*.5
        predicted=row['speaker'];confusion[truth,predicted or 'unknown']+=1
        if predicted is None: unknown+=1
        else: named+=1;correct+=int(predicted==truth)
        mixed+=int(len([v for v in overlap.values() if v>0])>1)
        details.append({**row,'reference':truth,'reference_overlap_seconds':overlap,'mixed_reference':len([v for v in overlap.values() if v>0])>1,'reference_ambiguous':ambiguous,'correct':predicted==truth})
    return {'windows':len(rows),'named_windows':named,'correct_named':correct,'named_accuracy':correct/named if named else None,'unknown_ratio':unknown/len(rows) if rows else None,'mixed_reference_windows':mixed,'confusion':[{'reference':a,'prediction':b,'count':v} for (a,b),v in sorted(confusion.items())],'details':details}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('speakers');parser.add_argument('--reference',required=True);parser.add_argument('--default',required=True);parser.add_argument('--start',type=float,default=0.);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=evaluate(json.loads(Path(args.speakers).read_text()),json.loads(Path(args.reference).read_text()),args.default,args.start)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
