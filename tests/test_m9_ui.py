from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from jevtells.ui.params import cli_args, intervals_from_drag, normalize, output_path, safe_path, source_rect
from jevtells.ui import server
from jevtells.ui.jobs import Jobs, redact


def params(**kwargs):
    return dict(input='samples/interview.mov', speaker='Tim', target=[[1, 0]], **kwargs)


def test_mapping_uses_cli_parser(tmp_path, monkeypatch):
    from jevtells.cli import _arguments
    p = params(start=16.3, duration=34.8, blur=[[15,15,250,60]], others_speaking=[[2.4,4.9]], subtitles='off')
    args = cli_args(p, tmp_path)
    monkeypatch.setattr('sys.argv', ['jevtells', *args])
    result = _arguments()
    assert result.start == '16.3' and result.duration == 34.8
    assert result.target == ['1@0'] and result.face_identity
    assert result.blur == ['15,15,250,60']
    assert result.others_speaking == ['2.4-4.9']
    assert result.output == str(tmp_path/'work/interview_s16.3_d34.8/output.mp4')
    assert '--duration' not in cli_args(params(), tmp_path)  # full source, no 30s default
    assert cli_args(params(), tmp_path, 'prepare')[-2:] == ['--until', 'segment']
    assert cli_args(params(), tmp_path, render_only=True)[-2:] == ['--from', 'render']


def test_mapping_auto_and_description(tmp_path):
    args = cli_args(params(start='auto', target_desc='white shirt'), tmp_path)
    assert '--target' not in args and '--target-desc' in args
    assert args[args.index('--start')+1] == 'auto'
    for p in [params(start=float('nan')), params(blur=[[0,0,0,1]]), params(lang='xx'), {**params(), 'key':'secret'}]:
        with pytest.raises((ValueError, TypeError)):
            cli_args(p, tmp_path)


@pytest.mark.parametrize('value', ['../.env', 'samples/../../.env', '/etc/passwd', 'work/../samples/x', 'samples/a\\b', '.env'])
def test_path_traversal(tmp_path, value):
    with pytest.raises(ValueError):
        safe_path(tmp_path, value)


def test_symlink_and_output_escape(tmp_path):
    (tmp_path/'samples').mkdir()
    (tmp_path/'samples/link').symlink_to('/etc')
    with pytest.raises(ValueError):
        safe_path(tmp_path, 'samples/link/passwd', 'samples')
    with pytest.raises(ValueError):
        output_path(params(output_dir='../escape'), tmp_path)


def test_intervals_and_rectangles():
    assert intervals_from_drag(80, 20, 100, 50) == [[10, 40]]
    assert intervals_from_drag(20,20,100,50) == []
    assert source_rect([10,10,100,30], [640,360], [1280,720]) == [20,20,200,60]
    assert source_rect([110,40,-100,-30], [640,360], [1280,720]) == [20,20,200,60]
    assert source_rect([-10,-10,1000,1000], [640,360], [1280,720]) == [0,0,1280,720]
    p = normalize(params(others_speaking=[[4,8],[2,5],[10,12]]))
    assert p['others_speaking'] == [[2,8],[10,12]]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'checks', lambda _: [{'name':'OPENROUTER_API_KEY','ok':False,'detail':'','repair':'Set key'}])
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    app = server.create_app(tmp_path)
    app.testing = True
    client = app.test_client()
    client.environ_base['HTTP_HOST'] = '127.0.0.1'
    # Werkzeug's base_url controls Host independently of environ_base.
    original = client.open
    def opened(*a, **kw):
        kw.setdefault('base_url', 'http://127.0.0.1')
        return original(*a, **kw)
    monkeypatch.setattr(client, 'open', opened)
    return client


def token(client):
    return {'X-JevTells-Token': client.get('/api/bootstrap').json['token']}


def test_host_origin_and_csrf(client):
    assert client.get('/').status_code == 200
    assert client.get('/', base_url='http://evil.test').status_code == 403
    assert client.post('/api/cancel').status_code == 403
    assert client.post('/api/cancel', headers={**token(client), 'Origin':'https://evil.test'}).status_code == 403
    assert client.post('/api/cancel', headers=token(client)).status_code == 200


def test_key_is_write_only_and_not_in_errors_logs_or_media(client, tmp_path, caplog):
    key = 'sk-or-private-M9-test-key'
    headers = token(client)
    response = client.post('/api/settings', json={'key':key}, headers=headers)
    assert response.status_code == 200 and key not in response.text
    assert key in (tmp_path/'.env').read_text()
    assert (tmp_path/'.env').stat().st_mode & 0o777 == 0o600
    assert key not in client.get('/api/settings').text
    assert client.get('/media/work/.env').status_code == 404
    assert key not in client.post('/api/settings', json={'key':key+'\ninvalid'}, headers=headers).text
    assert key not in caplog.text
    assert key not in redact('failed Bearer '+key, tmp_path)


def test_upload_rejects_before_write_and_media_range(client, tmp_path, monkeypatch):
    headers = token(client)
    assert client.post('/api/upload', data=b'x', headers={**headers,'X-Filename':'test.exe'}).status_code == 415
    client.application.config['MAX_CONTENT_LENGTH'] = 8
    assert client.post('/api/upload', data=b'0123456789', headers={**headers,'X-Filename':'test.mp4'}).status_code == 413
    assert list((tmp_path/'samples').iterdir()) == []
    client.application.config['MAX_CONTENT_LENGTH'] = 1000
    monkeypatch.setattr(server, 'media_info', lambda _: {'width':10,'height':10,'duration':1})
    response = client.post('/api/upload', data=b'0123456789', headers={**headers,'X-Filename':'测试.mov'})
    assert response.status_code == 200
    path = response.json['path']
    assert path.startswith('samples/video_') and path.endswith('.mov')
    result = client.get('/media/'+path, headers={'Range':'bytes=2-5'})
    assert result.status_code == 206 and result.data == b'2345'


def test_single_job_and_cancel(client):
    jobs = client.application.extensions['jobs']
    jobs.job['status'] = 'running'
    with pytest.raises(RuntimeError, match='busy'):
        jobs.start([], None, 'run')
    assert client.post('/api/settings', json={'lang':'en'}, headers=token(client)).status_code == 409
    jobs.cancel()
    assert jobs.job['status'] == 'cancelling'


def test_loopback_binding(monkeypatch):
    fake = Mock()
    fake.serve_forever.side_effect = KeyboardInterrupt
    create = Mock(return_value=Mock(extensions={'jobs':Mock()}))
    make = Mock(return_value=fake)
    monkeypatch.setattr(server, 'create_app', create)
    monkeypatch.setattr(server, 'make_server', make)
    monkeypatch.setattr(server.threading, 'Timer', Mock())
    server.serve(7861)
    assert make.call_args.args[:2] == ('127.0.0.1',7861)
    fake.server_close.assert_called_once()
    with pytest.raises(ValueError):
        server.serve(0)


def test_locale_files_and_json_headers(client):
    for lang in ('zh','en'):
        response = client.get('/static/locales/'+lang+'.json')
        assert response.status_code == 200
        assert len(response.json['steps']) == 5
        assert isinstance(response.json['stages'], dict)
    assert client.get('/static/locales/zh.json').json.keys() == client.get('/static/locales/en.json').json.keys()
    assert "frame-ancestors 'none'" in client.get('/').headers['Content-Security-Policy']


def test_timeline_and_coordinate_http_use_shared_mapping(client, tmp_path):
    headers = token(client)
    rect = [12,15,100,40]
    data={'rect':rect, 'display':[640,360], 'source':[1280,720]}
    assert client.post('/api/coordinates',json=data,headers=headers).json['rect'] == source_rect(**data)
    data={'kind':'timeline','a':100,'b':20,'width':200,'duration':50}
    assert client.post('/api/coordinates',json=data,headers=headers).json['intervals'] == [[5,25]]


def test_auto_render_uses_resolved_metadata_without_description_api(tmp_path):
    p = params(start='auto', target_desc='white shirt')
    output=output_path(p,tmp_path);output.mkdir(parents=True)
    (output/'run_meta.json').write_text(json.dumps({'parameters':{'start':16.3,'target':['1@0'], 'others_speaking':['2.4-4.9']}}))
    args=cli_args(p,tmp_path,render_only=True)
    assert args[args.index('--start')+1]=='16.3'
    assert '--target-desc' not in args and args[args.index('--target')+1]=='1@0'
    assert args[-2:]==['--from','render']


def test_saved_key_cannot_leak_through_echoed_parameters(client):
    key='sk-or-this-is-test-secret'
    headers=token(client)
    assert client.post('/api/settings',json={'key':key},headers=headers).status_code==200
    response=client.post('/api/command',json={'params':params(title=key)},headers=headers)
    assert response.status_code==200 and key not in response.text


def test_real_subprocess_logs_redaction_and_cancel(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    key='sk-or-process-test-secret'
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY='+key+'\n')
    original=subprocess.Popen
    def child(_args, **kwargs):
        return original([sys.executable,'-u','-c',f"import time; print('UI_STAGE pose',flush=True); print('{key}',flush=True); time.sleep(60)"],**kwargs)
    monkeypatch.setattr('jevtells.ui.jobs.subprocess.Popen',child)
    jobs=Jobs(tmp_path)
    jobs.start([],None,'people')
    deadline=time.monotonic()+8
    while not any('[redacted]' in row for row in jobs.snapshot()['logs']) and time.monotonic()<deadline:
        time.sleep(.02)
    snapshot=jobs.snapshot()
    assert snapshot['stage']=='pose' and key not in json.dumps(snapshot)
    assert any('[redacted]' in row for row in snapshot['logs'])
    jobs.cancel()
    jobs.worker.join(timeout=5)
    assert jobs.snapshot()['status']=='cancelled'
    assert jobs.process is None


def test_person_boxes_scale_from_normalized_clip_to_source(client, tmp_path, monkeypatch):
    out=tmp_path/'work/interview_s0_dauto';out.mkdir()
    (out/'detections.npz').touch()
    monkeypatch.setattr(server.people,'run',lambda *a,**k:[{'number':1,'box':[100,50,500,360],'time':0}])
    monkeypatch.setattr(server,'media_info',lambda p:{'width':3840,'height':2160} if p.name=='interview.mov' else {'width':1920,'height':1080})
    result=client.get('/api/people?input=samples/interview.mov')
    assert result.status_code==200 and result.json[0]['box']==[200,100,1000,720]


def test_interval_clamps_outside_drag():
    assert intervals_from_drag(-20,130,100,50)==[[0,50]]


def test_success_marker_is_ready_when_task_completes(tmp_path, monkeypatch):
    import subprocess
    import sys
    original=subprocess.Popen
    monkeypatch.setattr('jevtells.ui.jobs.subprocess.Popen',lambda _args,**kwargs:original([sys.executable,'-c','print("UI_STAGE render")'],**kwargs))
    output=tmp_path/'work/result';output.mkdir(parents=True)
    saved=json.dumps(params())
    (output/'ui_params.json').write_text(saved)
    jobs=Jobs(tmp_path)
    jobs.start([],output,'run')
    jobs.worker.join(timeout=5)
    assert jobs.snapshot()['status']=='complete'
    assert (output/'ui_complete_params.json').read_text()==saved


def test_equivalent_command_exposes_render_only_mode(client, tmp_path):
    p=normalize(params())
    output=output_path(p,tmp_path);output.mkdir(parents=True)
    (output/'ui_complete_params.json').write_text(json.dumps(p))
    (output/'narration.json').write_text('{}')
    changed={**p,'subtitles':'off'}
    result=client.post('/api/command',json={'params':changed},headers=token(client)).json
    assert result['args'][-2:]==['--from','render']
    changed['others_speaking']=[[0,1]]
    result=client.post('/api/command',json={'params':changed},headers=token(client)).json
    assert '--from' not in result['args']


def test_media_symlink_cannot_escape_into_home_root(client, tmp_path):
    (tmp_path/'private.mp4').write_bytes(b'private')
    (tmp_path/'samples/alias.mp4').symlink_to(tmp_path/'private.mp4')
    assert client.get('/media/samples/alias.mp4').status_code==400
    with pytest.raises(ValueError):
        safe_path(tmp_path,'samples/alias.mp4')
