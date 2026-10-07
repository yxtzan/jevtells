"""Opt-in persistent response accounting for bounded validation runs."""
from pathlib import Path
import json,os


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def before_request():
    configured=os.environ.get('JEVTELLS_API_LEDGER')
    if not configured: return
    rows=_rows(Path(configured))
    if any(row.get('cost') is None for row in rows): raise RuntimeError('API budget blocked: a response has unknown cost; reconcile ledger before continuing')
    total=sum(row['cost'] for row in rows)
    reserve=float(os.environ.get('JEVTELLS_API_RESERVE','0.05'))
    limit=float(os.environ.get('JEVTELLS_API_LIMIT','0.30'))
    if total+reserve>limit: raise RuntimeError(f'API budget exhausted: ${total:.8f} + ${reserve:.2f} reserve exceeds ${limit:.2f}')


def record(event):
    configured=os.environ.get('JEVTELLS_API_LEDGER')
    if not configured: return
    path=Path(configured);path.parent.mkdir(parents=True,exist_ok=True)
    rows=_rows(path);total=sum(float(row.get('cost') or 0) for row in rows)+float(event.get('cost') or 0)
    receipt={k:event.get(k) for k in ('status','error','cost','cost_reason','model','attempt')}
    receipt.update(sequence=len(rows)+1,total_cost=total)
    with path.open('a') as handle: handle.write(json.dumps(receipt,ensure_ascii=False)+'\n')
    progress=os.environ.get('JEVTELLS_API_PROGRESS')
    if progress:
        progress_path=Path(progress);text=progress_path.read_text();prefix=text.split('## API 费用累计')[0]
        progress_path.write_text(prefix+f'## API 费用累计\n\n- 上限 $0.30；已记录 {len(rows)+1} 次响应，已知累计 ${total:.9f}；未知费用 {sum(row.get("cost") is None for row in rows)+int(event.get("cost") is None)} 条。\n- 账本：`work/M7/api_ledger.jsonl`，每次响应即时更新。\n')
