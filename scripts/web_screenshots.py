"""Screenshots of the browser version, driven through headless Chrome (DevTools protocol).

    python scripts/web_screenshots.py            # builds _site, serves it, writes docs/figures/web_*.png

Needs Google Chrome and `pip install websocket-client`.  Each shot runs a few lines of JS in the
page (the same clicks a user would make) and waits until the page is idle.
"""
from __future__ import annotations

import base64
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
CHROME = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                           "/usr/bin/google-chrome", "/usr/bin/chromium")
               if os.path.exists(p)), None)
W, H = 1680, 1000


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CDP:
    def __init__(self, ws_url):
        import websocket
        self.ws = websocket.create_connection(ws_url, timeout=600, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr, timeout=180):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True,
                      timeout=timeout * 1000)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r.get("result", {}).get("value")

    def shot(self, name):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        (OUT / name).write_bytes(base64.b64decode(data))
        print("saved", name)


WAIT_IDLE = """new Promise(r => { const t0 = performance.now();
  const f = () => { const busy = !document.getElementById('busy3d').hidden || !document.getElementById('busy-slice').hidden
      || document.getElementById('st-head').textContent.includes('computing');
    (!busy && window.__idle && window.__idle()) || performance.now() - t0 > %d ? setTimeout(r, 600) : setTimeout(f, 150); };
  setTimeout(f, 400); })"""


def main() -> int:
    if CHROME is None:
        print("Chrome not found")
        return 1
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_site import build
    site = build()
    port = free_port()
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=str(site))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dbg = free_port()
    prof = tempfile.mkdtemp()
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={dbg}", f"--user-data-dir={prof}",
                               f"--window-size={W},{H}", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
                               "--no-first-run", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg}/json"))
                break
            except Exception:
                time.sleep(0.2)
        page = next(t for t in tabs if t["type"] == "page")
        c = CDP(page["webSocketDebuggerUrl"])
        c.call("Page.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        c.call("Page.navigate", url=f"http://127.0.0.1:{port}/index.html")
        t0 = time.time()
        c.js("new Promise(r => { const f = () => document.getElementById('loading').classList.contains('done') ? r() : setTimeout(f, 200); f(); })")
        c.js(WAIT_IDLE % 60000)
        print(f"loaded in {time.time() - t0:.1f} s:", c.js("document.getElementById('sb-info').textContent"))
        c.shot("web_3d.png")
        # the Σ-map tab (computes the default lens map on the first visit)
        c.js("document.querySelector('.tab[data-tab=\"sigma\"]').click(); 1")
        c.js("new Promise(r => { const f = () => document.getElementById('sigma-note').textContent.startsWith('done') ? setTimeout(r, 1500) : setTimeout(f, 250); f(); })")
        print("sigma:", c.js("document.getElementById('sigma-note').textContent"))
        c.shot("web_sigma.png")
        c.js("document.getElementById('sg-mode').value = 'mu'; document.getElementById('sg-mode').dispatchEvent(new Event('change')); 1")
        time.sleep(4)
        c.shot("web_sigma_mu.png")
        # back to 3D, with the 2D slice
        c.js("document.querySelector('.tab[data-tab=\"view3d\"]').click(); document.getElementById('btn-slice').click(); 1")
        c.js(WAIT_IDLE % 30000)
        time.sleep(2)
        c.shot("web_slice.png")
        # a path across the lens edge, scrubbed past the edge
        c.js("""(async () => { const $ = (i) => document.getElementById(i);
          $('btn-slice').click(); $('btn-path').click(); $('lock').click();
          const set = async (n, v) => { const e = $('p-' + n); e.value = v; e.dispatchEvent(new Event('change')); await new Promise(r => setTimeout(r, 1500)); };
          await set('gamma', -1.2); await set('kappa', 0.5); $('pa-add').click();
          await set('gamma', 0.45); await set('kappa', 3.3); $('pa-add').click();
          await set('gamma', 0.3); await set('kappa', 4.4); $('pa-add').click();
          await new Promise(r => { const f = () => / frames are/.test($('pa-events').textContent) ? r() : setTimeout(f, 200); f(); });
          return 1; })()""")
        print("path:", c.js("document.getElementById('pa-events').textContent.slice(0, 300)"),
              c.js("[...document.querySelectorAll('#pa-table tbody tr')].map(t => t.textContent).join(' | ')"),
              "lock", c.js("document.getElementById('lock').checked"))
        c.js("(document.querySelector('#pa-events a[data-i]') || document.getElementById('pa-last')).click(); 1")
        c.js(WAIT_IDLE % 30000)
        time.sleep(2)
        c.shot("web_path.png")
        c.js("document.querySelector('.tab[data-tab=\"sigma\"]').click(); document.getElementById('sg-mode').value='global'; document.getElementById('sg-mode').dispatchEvent(new Event('change')); 1")
        time.sleep(5)
        c.shot("web_path_sigma.png")
        # light theme
        c.js("document.querySelector('.tab[data-tab=\"view3d\"]').click(); document.getElementById('btn-theme').click(); 1")
        c.js(WAIT_IDLE % 30000)
        time.sleep(3)
        c.shot("web_light.png")
        errs = c.js("JSON.stringify(window.__errors || [])")
        print("page errors:", errs)
    finally:
        chrome.terminate()
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
