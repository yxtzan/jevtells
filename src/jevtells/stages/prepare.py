from pathlib import Path
import subprocess,json

def run(inp:Path,out:Path,force=False,start=0,duration=None):
 out.mkdir(parents=True,exist_ok=True); clip=out/'clip.mp4'; wav=out/'audio.wav'
 if force or not clip.exists():
  cmd=['ffmpeg','-y','-ss',str(start),'-i',str(inp)]
  if duration: cmd+=['-t',str(duration)]
  cmd+=['-c','copy',str(clip)]
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 if force or not wav.exists(): subprocess.run(['ffmpeg','-y','-i',str(clip),'-ar','16000','-ac','1',str(wav)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 return clip,wav
