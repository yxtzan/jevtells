from types import SimpleNamespace

from jevtells import doctor, resources
from jevtells.config import load_config


def test_asset_search_precedence_and_absolute_override(tmp_path, monkeypatch):
    configured=tmp_path/'configured';current=tmp_path/'current';home=tmp_path/'home'
    for root in (configured,current,home/'.jevtells'):
        (root/'models').mkdir(parents=True)
        (root/'models/pose.task').write_text('asset')
    monkeypatch.setenv('JEVTELLS_HOME',str(configured))
    monkeypatch.chdir(current)
    monkeypatch.setattr(resources.Path,'home',classmethod(lambda cls: home))
    assert resources.asset_path('models/pose.task')==configured/'models/pose.task'
    (configured/'models/pose.task').unlink()
    assert resources.asset_path('models/pose.task')==current/'models/pose.task'
    (current/'models/pose.task').unlink()
    assert resources.asset_path('models/pose.task')==home/'.jevtells/models/pose.task'
    assert resources.asset_path(tmp_path/'explicit.task')==tmp_path/'explicit.task'


def test_doctor_does_not_write_or_reveal_key_and_rejects_unusable_encoders(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('JEVTELLS_HOME',str(tmp_path))
    monkeypatch.setenv('OPENROUTER_API_KEY','private-test-value')
    monkeypatch.setattr(resources.Path,'home',classmethod(lambda cls: tmp_path/'home'))
    monkeypatch.setattr(doctor.shutil,'which',lambda _: '/test/ffmpeg')
    def execute(command, **kwargs):
        if '-encoders' in command:
            return SimpleNamespace(stdout='libx264 h264_videotoolbox')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(doctor.subprocess,'run',execute)
    assert doctor.run()==1
    output=capsys.readouterr().out
    assert 'private-test-value' not in output
    assert 'OPENROUTER_API_KEY: 已设置' in output
    assert 'FAIL ffmpeg' in output and 'Fix:' in output
    assert list(tmp_path.iterdir())==[]


def test_doctor_key_file_status_and_successful_encoder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('JEVTELLS_HOME',str(tmp_path))
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY=file-private-test-value\n')
    monkeypatch.setattr(doctor.shutil,'which',lambda _: '/test/ffmpeg')
    monkeypatch.setattr(doctor.subprocess,'run',lambda command,**kwargs: SimpleNamespace(stdout='libx264',returncode=0))
    config=load_config()
    for relative in [*config['render']['fonts'].values(),*('models/'+value for value in config['models'].values())]:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('test fixture')
    before={str(path.relative_to(tmp_path)):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert doctor.run(config)==0
    assert 'file-private-test-value' not in capsys.readouterr().out
    assert before=={str(path.relative_to(tmp_path)):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    (tmp_path/'.env').unlink()
    assert doctor.run(config)==1
    assert 'OPENROUTER_API_KEY: 未设置' in capsys.readouterr().out


def test_default_package_resources_and_external_config(tmp_path, monkeypatch):
    for name in ('config/default.yaml','config/narrate_prompt.md','config/jev_questions.yaml','i18n/zh.yaml'):
        assert resources.data_path(name).is_file()
    monkeypatch.chdir(tmp_path)
    (tmp_path/'custom.yaml').write_text('render:\n  far_shoulder_ratio: 0.123\n')
    assert load_config('custom.yaml')['render']['far_shoulder_ratio']==0.123
