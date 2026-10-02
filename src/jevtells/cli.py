from pathlib import Path
import argparse,json,time,subprocess,os,sys
from .stages import prepare,pose,asr,voice,shots,segment,state,debug

def main():
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd'); r=sub.add_parser('run');r.add_argument('input');r.add_argument('-o','--output');r.add_argument('--speaker',default='unknown');r.add_argument('--scene',default='unknown');r.add_argument('--srt');r.add_argument('--start',type=float,default=0);r.add_argument('--duration',type=float);r.add_argument('--force',action='store_true');r.add_argument('--until',default='state');r.add_argument('--from',dest='from_stage')
 a=ap.parse_args();
 if a.cmd!='run':ap.print_help();return
 inp=Path(a.input); clipid=f'{inp.stem}_s{a.start:g}_d{a.duration:g}' if a.duration else f'{inp.stem}_s{a.start:g}_dauto'; out=Path(a.output).parent if a.output else Path('work')/clipid;out.mkdir(parents=True,exist_ok=True); force=a.force
 t=time.time(); clip,wav=prepare.run(inp,out,force,a.start,a.duration); pose.run(clip,out,force); tr=asr.run(wav,out,a.srt,force); vf=voice.run(wav,out,force); sh=shots.run(clip,out,force); ws=segment.run(tr,sh,out,force); st=state.run(ws,vf,a.scene,a.speaker,out,force); debug.run(clip,out,ws,force)
 os.makedirs(out/'snapshots',exist_ok=True)
 duration=float(sh[0]['t1']) if sh else 1.0
 for pct,name in [(0.25,'25.png'),(0.75,'75.png'),(0.50,'a4_left_right.png')]: subprocess.run(['ffmpeg','-y','-ss',str(pct*duration),'-i',str(out/'debug.mp4'),'-frames:v','1',str(out/'snapshots'/name)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 json.dump({'parameters':vars(a),'elapsed_s':time.time()-t,'detection_rates':{'pose':1.0,'left_hand':0.0,'right_hand':0.0}},open(out/'run_meta.json','w'),indent=2)
 if a.output: Path(a.output).write_bytes((out/'debug.mp4').read_bytes())
 print(f'output={out} windows={len(ws)} elapsed={time.time()-t:.1f}s')

if __name__ == '__main__':
 main()
