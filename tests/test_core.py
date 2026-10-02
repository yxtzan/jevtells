from jevtells.stages.segment import run

def test_windows(tmp_path):
 d=run({'segments':[{'t0':0,'t1':3,'text':'hello'},{'t0':4,'t1':8,'text':'world'}]},[],tmp_path,True);assert len(d)==2
