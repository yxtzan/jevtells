import json

def run(wav,out,srt=None,force=False):
 p=out/'transcript.json'
 if p.exists() and not force:return json.load(open(p))
 seg=[]
 if srt:
  import re
  txt=open(srt).read(); blocks=re.split(r'\n\s*\n',txt)
  for b in blocks:
   m=re.search(r'(\d\d):(\d\d):(\d\d),(\d+) --> (\d\d):(\d\d):(\d\d),(\d+)\n(.+)',b,re.S)
   if m: seg.append({'t0':int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3))+int(m.group(4))/1000,'t1':int(m.group(5))*3600+int(m.group(6))*60+int(m.group(7))+int(m.group(8))/1000,'text':m.group(9).strip(),'words':[]})
 else: seg=[]
 d={'language':'unknown','segments':seg};json.dump(d,open(p,'w'),ensure_ascii=False,indent=2);return d
