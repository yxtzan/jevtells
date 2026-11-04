"""Real Chrome screenshots and browser interaction checks (optional Playwright)."""
from __future__ import annotations
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
SCREENS=ROOT/'work/M9/screens'


def main() -> None:
    SCREENS.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:7860')
        page.wait_for_function('state.token && Object.keys(state.dict).length > 0')
        page.wait_for_function("async()=>!['running','cancelling'].includes((await api('/api/jobs')).status)",timeout=900000,polling=2000)
        params=json.loads((ROOT/'work/M9/validation_params.json').read_text())
        output='work/M9/http/tim_interview_s16.3_dauto'
        # Reuse the exact real preprocessing output, without mutating its parameters.
        page.evaluate('async ([p,o]) => {await restore(p,o);}',[params,output])
        page.wait_for_function("document.fonts.status === 'loaded'")
        for step in range(1,6):
            page.evaluate('async step=>await go(step)',step)
            page.wait_for_timeout(800)
            page.screenshot(path=str(SCREENS/f'0{step}.png'),full_page=True)
        page.evaluate("async()=>await go('Settings')")
        page.screenshot(path=str(SCREENS/'06-settings.png'),full_page=True)
        page.evaluate("async()=>await go('History')")
        page.wait_for_timeout(800)
        page.screenshot(path=str(SCREENS/'07-history.png'),full_page=False)
        page.evaluate("async()=>{await language('en');await go(4);}")
        page.screenshot(path=str(SCREENS/'08-english.png'),full_page=True)
        # Real drag → shared HTTP coordinate converter → source pixels.
        page.evaluate('async()=>await go(3)')
        before=page.evaluate('state.blur.length')
        bounds=page.locator('#blurFrame').bounding_box()
        page.mouse.move(bounds['x']+20,bounds['y']+20)
        page.mouse.down();page.mouse.move(bounds['x']+120,bounds['y']+60,steps=8);page.mouse.up()
        page.wait_for_function('(count)=>state.blur.length === count+1',arg=before)
        drag=page.evaluate('state.blur.at(-1)')
        assert drag[2]>100 and drag[3]>40
        page.evaluate('async()=>{state.blur.pop();await go(4);}')
        assert page.locator('#cues button').count()>0
        subtitles=page.locator('#cues button').count()
        prior=page.evaluate('JSON.stringify(state.others)')
        page.locator('#cues button').first.click()
        assert page.evaluate('JSON.stringify(state.others)')!=prior
        page.evaluate('state.others='+json.dumps(params['others_speaking']))
        # Offline UI language switching, no CDN or network resources.
        requests=[]
        page.on('request',lambda req:requests.append(req.url))
        page.reload();page.wait_for_function('state.token && Object.keys(state.dict).length > 0')
        assert not [u for u in requests if not u.startswith('http://127.0.0.1:7860')]
        assert not errors, errors
        (ROOT/'work/M9/browser_review.json').write_text(json.dumps({'browser':browser.version,'errors':errors,'screens':sorted(p.name for p in SCREENS.glob('*.png')),'drag_source_rect':drag,'timeline_subtitles':subtitles,'external_requests':[]},indent=2))
        browser.close()
        print('Real browser review passed')


if __name__=='__main__':main()
