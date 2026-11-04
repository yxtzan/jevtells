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
