// Single-file dashboard served at "/"
// Live MJPEG + WASD via WebSocket (ws://<host>:81/). Falls back to /cmd HTTP.

#include <Arduino.h>

extern const char DASHBOARD_HTML[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ESP32 RC Control</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;padding:16px}
  h1{margin:0 0 8px;font-size:18px;font-weight:500}
  #stream{width:min(90vw,800px);border:2px solid #333;border-radius:8px;background:#000}
  .pad{display:grid;grid-template-columns:60px 60px 60px;gap:6px;margin-top:16px}
  .pad div{height:60px;background:#222;border:1px solid #444;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:600;color:#888}
  .pad .on{background:#2d7;color:#000;border-color:#2d7}
  .pad .e{visibility:hidden}
  #log{margin-top:12px;font-family:ui-monospace,monospace;font-size:12px;color:#888;min-height:1.4em}
  #status{font-size:12px;color:#888;margin-bottom:8px}
  #wsStatus{font-size:11px;color:#888;margin-top:4px}
  a{color:#5af}
</style></head>
<body>
<h1>ESP32-CAM Control &mdash; focus this page, use W A S D</h1>
<div id="status">stream loading...</div>
<img id="stream" src="/stream" onload="document.getElementById('status').textContent='stream live'"
     onerror="document.getElementById('status').textContent='stream error'">
<div class="pad">
  <div class="e"></div><div id="kW">W</div><div class="e"></div>
  <div id="kA">A</div><div id="kS">S</div><div id="kD">D</div>
</div>
<div id="log">ready</div>
<div id="wsStatus">ws: connecting...</div>
<div id="stats" style="margin-top:10px;font-family:ui-monospace,monospace;font-size:12px;color:#7df;">stats: --</div>
<div style="margin-top:12px;font-size:12px;">
  presets:
  <button onclick="preset('ulow')">ultra-low-lat</button>
  <button onclick="preset('low')">low-lat</button>
  <button onclick="preset('bal')">balanced</button>
  <button onclick="preset('hi')">quality</button>
  <button onclick="autoTune()">auto-tune</button>
</div>
<div style="margin-top:10px;font-size:12px;">
  resolution:
  <select id="resSel" onchange="setRes(this.value)">
    <option value="qqvga">QQVGA 160x120</option>
    <option value="qvga" selected>QVGA 320x240</option>
    <option value="cif">CIF 400x296</option>
    <option value="vga">VGA 640x480</option>
    <option value="svga">SVGA 800x600</option>
    <option value="xga">XGA 1024x768</option>
    <option value="hd">HD 1280x720</option>
    <option value="uxga">UXGA 1600x1200</option>
  </select>
  quality: <input id="qIn" type="number" min="4" max="40" value="15" style="width:50px" onchange="setRes(document.getElementById('resSel').value)">
</div>
<div style="margin-top:8px;font-size:11px;"><a href="/update">OTA firmware update</a></div>
<script>
const keys = {w:false,a:false,s:false,d:false};
const log = document.getElementById('log');
const wsStatus = document.getElementById('wsStatus');

let socket = null;
let wsReady = false;

function connectWS(){
  const host = location.hostname;
  socket = new WebSocket(`ws://${host}:81/`);
  socket.onopen = () => { wsReady = true; wsStatus.textContent = 'ws: connected'; };
  socket.onclose = () => {
    wsReady = false; wsStatus.textContent = 'ws: disconnected — retrying...';
    setTimeout(connectWS, 1000);
  };
  socket.onerror = () => { wsStatus.textContent = 'ws: error — using HTTP fallback'; };
}
connectWS();

function setRes(size, q){
  q = q ?? document.getElementById('qIn').value;
  document.getElementById('resSel').value = size;
  document.getElementById('qIn').value = q;
  return fetch(`/res?size=${size}&q=${q}`).then(r => r.text()).then(t => {
    log.textContent = 'res: '+size+' q='+q+' → '+t;
    const img = document.getElementById('stream');
    img.src = '/stream?t='+Date.now();
  }).catch(e => log.textContent = 'res err: '+e);
}

const PRESETS = {
  ulow: {size:'qqvga', q:20},   // tiny, max fps, weak WiFi survival
  low:  {size:'qvga',  q:15},   // typical low-lat
  bal:  {size:'cif',   q:14},   // balanced
  hi:   {size:'vga',   q:12},   // quality, needs decent RSSI
};
function preset(name){
  const p = PRESETS[name];
  setRes(p.size, p.q);
}

// Auto-tune: cycle presets, measure fps after 3s settle, pick best.
async function autoTune(){
  log.textContent = 'auto-tune: testing...';
  let best = {name:'', fps:0};
  for (const [name, p] of Object.entries(PRESETS)) {
    await setRes(p.size, p.q);
    await new Promise(r => setTimeout(r, 3000));
    const s = await fetch('/stats').then(r => r.json());
    log.textContent = `auto-tune ${name}: ${s.fps} fps`;
    // Pick highest fps × resolution heuristic. fps × pixels.
    const score = s.fps;
    if (score > best.fps + 1) best = {name, fps: s.fps};
  }
  const p = PRESETS[best.name];
  await setRes(p.size, p.q);
  log.textContent = `auto-tune winner: ${best.name} (${best.fps} fps)`;
}

// Poll /stats every 1s.
setInterval(() => {
  fetch('/stats').then(r => r.json()).then(s => {
    document.getElementById('stats').textContent =
      `rssi ${s.rssi} dBm  |  ${s.fps} fps  |  ${(s.kbps/1000).toFixed(1)} Mbps  |  heap ${(s.heap/1024).toFixed(0)}KB  |  psram ${(s.psram/1024).toFixed(0)}KB`;
  }).catch(()=>{});
}, 1000);

function send(k, down){
  const msg = `${k}${down?1:0}`;
  if (wsReady) {
    try { socket.send(msg); return; } catch(e) { wsReady = false; }
  }
  // Fallback: HTTP cmd if WS not up.
  fetch(`/cmd?k=${k}&s=${down?1:0}`).catch(e => log.textContent = 'cmd err: '+e);
}
function paint(){
  for (const k of ['w','a','s','d']) {
    document.getElementById('k'+k.toUpperCase()).className = keys[k]?'on':'';
  }
  log.textContent = `W=${+keys.w} A=${+keys.a} S=${+keys.s} D=${+keys.d}`;
}
addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (!(k in keys) || keys[k]) return;
  keys[k] = true; send(k, true); paint();
});
addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  if (!(k in keys)) return;
  keys[k] = false; send(k, false); paint();
});
addEventListener('blur', () => {
  for (const k of ['w','a','s','d']) if (keys[k]) { keys[k]=false; send(k,false); }
  paint();
});
</script></body></html>
)HTML";
