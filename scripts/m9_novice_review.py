"""A browser walkthrough using a fresh four-second clip and no existing cache.

Fixture: samples/m9_novice.mov, made from a short interview section.
Every interaction uses the visible controls and their normal HTTP handlers.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    started=time.monotonic()
    actions=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=os.environ.get('JEVTELLS_CHROME','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1080})
        errors=[]
        def until(expression: str, timeout: int = 180000) -> None:
            deadline=time.monotonic()+timeout/1000
            while not page.evaluate(expression):
                if time.monotonic()>deadline: raise TimeoutError(expression)
                page.wait_for_timeout(300)
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto(os.environ.get('JEVTELLS_UI','http://127.0.0.1:7860'))
        until('state.token && Array.isArray(state.dict.steps)')
        until("async()=>!['running','cancelling'].includes((await api('/api/jobs')).status)",timeout=900000)
        await_start=time.monotonic()
        def action(label):
            actions.append(label)
            print(label,flush=True)
        # Use the native drop handler with a real local video File object.
        transfer=page.evaluate_handle("""async()=>{const response=await fetch('/media/samples/m9_novice.mov');const blob=await response.blob();const data=new DataTransfer();data.items.add(new File([blob],'新手测试.mov',{type:'video/quicktime'}));return data;}""")
        page.dispatch_event('#drop','drop',{'dataTransfer':transfer});action('1. 拖入新视频')
        until("state.input.startsWith('samples/video_') && state.info?.duration>0")
        # Person detection starts by itself after the video is chosen.
        until("!!document.querySelector('#personBoxes .pbox')",timeout=180000)
        page.locator('#speaker').fill('Tim');action('2. 填写主角名字')
        page.locator('#personBoxes .pbox').first.click();action('3. 点击画面里的人')
        page.locator('#nextStep').click();action('4. 下一步：框水印')
        bounds=page.locator('#frame').bounding_box()
        page.mouse.move(bounds['x']+bounds['width']*.012,bounds['y']+bounds['height']*.021)
        page.mouse.down();page.mouse.move(bounds['x']+bounds['width']*.205,bounds['y']+bounds['height']*.104,steps=8);page.mouse.up();action('5. 拖框模糊水印')
        until('state.blur.length===1')
        page.locator('#nextStep').click();action('6. 下一步：标说话人')
        # Preparation runs in the background once the subject step is left.
        until("state.timeline?.windows.length>0",timeout=180000)
        page.locator('#generate').click();action('7. 生成成片')
        until("!!state.results && !document.getElementById('viewDone').hidden",timeout=240000)
        result=page.evaluate('({input:state.input, output:state.output, job:state.job, duration:state.info.duration, params:params(), subtitle_blocks:state.timeline.windows.length})')
        assert result['job']['status']=='complete'
        assert result['duration']<4.1
        assert not errors,errors
        page.screenshot(path=str(ROOT/'work/M9b/screens/novice-complete.png'),full_page=True)
        (ROOT/'work/M9b/novice_review.json').write_text(json.dumps({'actions':actions,'clicks':5,'text_entries':1,'drags':2,'interactions':len(actions),'elapsed_seconds':time.monotonic()-await_start,'errors':errors,'result':result,'findings':[]},ensure_ascii=False,indent=2))
        browser.close()
        print(f'Fresh upload to finished videos: {len(actions)} interactions',flush=True)


if __name__=='__main__':main()
