import cv2,subprocess,json

def run(clip,out,windows,force=False):
 p=out/'debug.mp4'
 if p.exists() and not force:return p
 cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(3));h=int(cap.get(4)); tmp=out/'debug_noaudio.mp4'; writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'avc1'),fps,(w,h)); i=0
 while True:
  ok,fr=cap.read()
  if not ok:break
  t=i/fps; win=next((x for x in windows if x['t0']<=t<=x['t1']),None)
  cv2.putText(fr,f't={t:.2f}  {win["id"] if win else ""}  shot=target',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),2)
  if win: cv2.putText(fr,win['subtitle'][:80],(20,h-25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  writer.write(fr);i+=1
 cap.release();writer.release();subprocess.run(['ffmpeg','-y','-i',str(tmp),'-i',str(clip),'-map','0:v','-map','1:a?','-c:v','copy','-c:a','aac','-shortest',str(p)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);tmp.unlink();return p
