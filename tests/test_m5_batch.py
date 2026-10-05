import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from batch_eval import alerts


def test_batch_flags_low_lock_fallback_collisions_and_rate_per_group():
    def row(identifier,group,rate,lock=1,fallback=0,collisions=0):
        return {'id':identifier,'group':group,'events_per_minute':rate,'target_lock_rate':lock,'narration_fallback_ratio':fallback,'layout_violation_frames':collisions}
    rows=[row('a','tune',10,.7,.5,3),row('b','tune',10),row('c','tune',100),row('h','holdout',1000)]
    alerts(rows,{})
    assert 'lock' in rows[0]['alerts'] and 'fallback' in rows[0]['alerts'] and 'collisions' in rows[0]['alerts']
    assert 'event rate' in rows[2]['alerts']
    assert rows[3]['alerts']==''
