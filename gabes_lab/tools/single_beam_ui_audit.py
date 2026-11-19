"""Exercise the real SABES single-beam profiler with the shared browser audit.

Outputs screenshots and a compact interaction receipt under runs/. The first
Run evaluates the actual atomic engine; camera/view navigation reuses fields.
"""
import asyncio
import json
from pathlib import Path
import tempfile
import shutil
import time

from gabes_lab.blocks.runner import start_run
from gabes_lab.blocks.io import write_json
from gabes_lab.tools.ui_audit import (CDP, DESKTOP, MOBILE, _center_js, find_browser,
    launch_browser, page_ws_url, screenshot, set_sidebar, spawn_streamlit, wait_ready)


async def click(cdp, expression):
    await cdp.js(expression+"?.scrollIntoView({block:'center'});")
    await asyncio.sleep(.15)
    point=await cdp.js(_center_js(expression))
    if not point:
        raise RuntimeError("Control not found: "+expression)
    await cdp.click(*point)
    await asyncio.sleep(.2)


def text_element(selector,text):
    return (f"[...document.querySelectorAll({json.dumps(selector)})].find(e => "
            f"(e.textContent.trim()==={json.dumps(text)} || e.querySelector('p')?.textContent.trim()==={json.dumps(text)}))")


async def select(cdp,label,value):
    expression=("[...document.querySelectorAll('[role=combobox]')].find(e => "
                f"(e.getAttribute('aria-label')||'').endsWith({json.dumps('. '+label)}))")
    await click(cdp,expression)
    await click(cdp,text_element('[role=option]',value))
    await wait_ready(cdp,timeout=180,min_wait=.2)


async def inspect(cdp):
    return await cdp.js("""(() => ({
      labels:[...document.querySelectorAll('[role=combobox],input')].map(e=>({tag:e.tagName,label:e.getAttribute('aria-label'),type:e.type,value:e.value})),
      buttons:[...document.querySelectorAll('button')].map(e=>e.textContent.trim()),
      exceptions:[...document.querySelectorAll('[data-testid=stException]')].map(e=>e.textContent),
      overflow:document.documentElement.scrollWidth>innerWidth,
      text:document.querySelector('[data-testid=stMain]')?.innerText
    }))()""")


async def run(cdp,url,run):
    await cdp.send('Page.enable')
    await cdp.send('Emulation.setDeviceMetricsOverride',**DESKTOP)
    await cdp.send('Page.navigate',url=url+'?app=sabes')
    await wait_ready(cdp,timeout=180,min_wait=1.)
    write_json(run.path('initial_dom.json'),await inspect(cdp))
    await select(cdp,'Instrument','Beam profiler')
    await click(cdp,text_element('label','Rb single-beam'))
    await wait_ready(cdp,timeout=180,min_wait=.2)
    await cdp.js(text_element('h6','Rb single-beam profiler')+".scrollIntoView({block:'start'});")
    await screenshot(cdp,run.path('controls_desktop.png'))
    start=time.monotonic()
    await click(cdp,text_element('button','Run single-beam calculation'))
    await wait_ready(cdp,timeout=300,min_wait=1.)
    elapsed=time.monotonic()-start
    result=await inspect(cdp)
    if 'Cell refinement passed' not in result['text'] or result['exceptions']:
        write_json(run.path('failed_dom.json'),result)
        raise RuntimeError('Real profiler calculation did not pass')
    await cdp.js("document.querySelector('[data-testid=stAlertContainer]')?.scrollIntoView({block:'start'});")
    # Scroll to the actual profiler result rather than global apparatus alerts.
    await cdp.js("[...document.querySelectorAll('[data-testid=stAlert]')].find(e=>e.textContent.includes('Cell refinement passed'))?.scrollIntoView({block:'start'});")
    await screenshot(cdp,run.path('result_desktop.png'))
    await click(cdp,text_element('label','2D intensity'))
    await wait_ready(cdp,timeout=30,min_wait=.2)
    await cdp.js("[...document.querySelectorAll('[data-testid=stAlert]')].find(e=>e.textContent.includes('Cell refinement passed'))?.scrollIntoView({block:'start'});")
    await screenshot(cdp,run.path('intensity_2d_desktop.png'))
    start=time.monotonic()
    entry="document.querySelector('input[aria-label=\"Camera distance after cell (mm)\"]')"
    await click(cdp,entry)
    await cdp.send('Input.dispatchKeyEvent',type='keyDown',key='a',code='KeyA',windowsVirtualKeyCode=65,modifiers=2)
    await cdp.send('Input.dispatchKeyEvent',type='keyUp',key='a',code='KeyA',windowsVirtualKeyCode=65,modifiers=2)
    await cdp.send('Input.insertText',text='700')
    await cdp.send('Input.dispatchKeyEvent',type='keyDown',key='Enter',code='Enter',windowsVirtualKeyCode=13)
    await cdp.send('Input.dispatchKeyEvent',type='keyUp',key='Enter',code='Enter',windowsVirtualKeyCode=13)
    # Commit the numeric edit on blur; an Enter key without a character
    # event is insufficient on some browser/Streamlit versions.
    await click(cdp,text_element('h6','Rb single-beam profiler'))
    await wait_ready(cdp,timeout=30,min_wait=1.2)
    camera_elapsed=time.monotonic()-start
    state=await inspect(cdp)
    if (state['exceptions'] or 'Cell refinement passed' not in state['text']
            or '0.431914 mm' in state['text']):
        raise RuntimeError('Camera navigation did not update the saved-field profile')
    await cdp.send('Emulation.setDeviceMetricsOverride',**MOBILE)
    await set_sidebar(cdp,False)
    await cdp.js(text_element('h6','Rb single-beam profiler')+".scrollIntoView({block:'start'});")
    await asyncio.sleep(.5)
    await screenshot(cdp,run.path('controls_mobile.png'))
    await cdp.js("[...document.querySelectorAll('[data-testid=stAlert]')].find(e=>e.textContent.includes('Cell refinement passed'))?.scrollIntoView({block:'start'});")
    await screenshot(cdp,run.path('result_mobile.png'))
    await click(cdp,text_element('button','Cell & laser'))
    await screenshot(cdp,run.path('conditions_mobile.png'))
    await cdp.send('Input.dispatchKeyEvent',type='keyDown',key='Escape',code='Escape',windowsVirtualKeyCode=27)
    await cdp.send('Input.dispatchKeyEvent',type='keyUp',key='Escape',code='Escape',windowsVirtualKeyCode=27)
    mobile=await inspect(cdp)
    await cdp.send('Emulation.setDeviceMetricsOverride',**DESKTOP)
    await select(cdp,'Input beam','Cell seed line')
    imported=await inspect(cdp)
    if imported['exceptions'] or 'Inputs changed' not in imported['text']:
        raise RuntimeError('Source change did not invalidate the old result')
    await cdp.js(text_element('h6','Rb single-beam profiler')+".scrollIntoView({block:'start'});")
    await screenshot(cdp,run.path('seed_input_desktop.png'))
    write_json(run.path('interaction.json'),dict(real_calculation_seconds=elapsed,
        camera_navigation_seconds=camera_elapsed,initial_result_passed=True,
        source_change_stale=True,mobile_overflow=mobile['overflow'],
        desktop=result,mobile=mobile,imported_source=imported))
    print(f'PASS: real calculation {elapsed:.2f}s, camera navigation {camera_elapsed:.2f}s',flush=True)


def main(argv=None):
    with start_run('ui_audit/single_beam_live') as record:
        app,url=spawn_streamlit()
        browser,profile,port=launch_browser(find_browser())
        print(str(record.dir),flush=True)
        async def execute():
            cdp=await CDP.connect(page_ws_url(port))
            try:
                await run(cdp,url,record)
            except BaseException:
                write_json(record.path("failed_dom.json"),await inspect(cdp))
                await screenshot(cdp,record.path("failure.png"))
                raise
            finally:
                await cdp.close()
        try:
            asyncio.run(execute())
        finally:
            app.terminate();browser.terminate()
            app.wait(timeout=20);browser.wait(timeout=20)
            # This temporary browser profile belongs to this audit process.
            target = Path(profile).resolve()
            temporary_root = Path(tempfile.gettempdir()).resolve()
            if target.parent != temporary_root or not target.name.startswith("gabes_audit_"):
                raise RuntimeError("unexpected browser profile cleanup scope")
            shutil.rmtree(target,ignore_errors=True)


if __name__=='__main__':
    main()
