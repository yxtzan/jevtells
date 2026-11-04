"""A browser walkthrough using a fresh four-second clip and no existing cache.

Fixture: samples/m9_novice.mov, made from a short interview section.
Every interaction uses the visible controls and their normal HTTP handlers.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    started=time.monotonic()
    actions=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1080})
        errors=[]
        def until(expression: str, timeout: int = 180000) -> None:
            deadline=time.monotonic()+timeout/1000
            while not page.evaluate(expression):
                if time.monotonic()>deadline: raise TimeoutError(expression)
                page.wait_for_timeout(300)
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto('http://127.0.0.1:7860')
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
        page.locator('#next').click();action('2. 下一步：选主角')
        page.locator('#speaker').fill('Tim');action('3. 填写主角名字')
        page.locator('#detect').click();action('4. 检测画面中的人')
        until("state.job?.mode==='people' && state.job?.status==='complete'",timeout=180000)
        page.locator('#personBoxes rect[role=button]').first.click();action('5. 点击人物框')
        page.locator('#next').click();action('6. 下一步：画面处理')
        bounds=page.locator('#blurFrame').bounding_box()
        page.mouse.move(bounds['x']+bounds['width']*.012,bounds['y']+bounds['height']*.021)
        page.mouse.down();page.mouse.move(bounds['x']+bounds['width']*.205,bounds['y']+bounds['height']*.104,steps=8);page.mouse.up();action('7. 拖框模糊水印')
        until('state.blur.length===1')
        page.locator('#next').click();action('8. 下一步：谁在说话')
        page.locator('#prepare').click();action('9. 运行预处理')
        until("state.job?.mode==='prepare' && state.job?.status==='complete' && state.timeline?.windows.length>0",timeout=180000)
        page.locator('#next').click();action('10. 下一步：生成')
        page.locator('#generate').click();action('11. 生成成片')
        until("state.job?.mode==='run' && state.job?.status==='complete' && !document.getElementById('results').hidden",timeout=240000)
        result=page.evaluate('({input:state.input, output:state.output, job:state.job, duration:state.info.duration, params:params(), subtitle_blocks:state.timeline.windows.length})')
        assert result['job']['status']=='complete'
        assert result['duration']<4.1
        assert not errors,errors
        page.screenshot(path=str(ROOT/'work/M9/screens/09-novice-complete.png'),full_page=True)
        (ROOT/'work/M9/novice_review.json').write_text(json.dumps({'actions':actions,'clicks':8,'text_entries':1,'drags':2,'interactions':11,'elapsed_seconds':time.monotonic()-await_start,'errors':errors,'result':result,'findings':['首次检测和预处理均需等待，阶段提示可解释等待。','人物框需要先检测才出现，按钮可见但仍需读一次说明。','字幕自动预选省去一次操作，仍应查看截图确认。','精确选取起点只靠滑块不便，后续可补数字时间输入。','本次为单人四秒新素材，复杂多机位素材需要额外锚点和他人发言标记。']},ensure_ascii=False,indent=2))
        browser.close()
        print('Fresh upload to finished videos: 11 interactions',flush=True)


if __name__=='__main__':main()
