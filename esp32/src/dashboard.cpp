// Single-file dashboard served at "/"
// Live MJPEG + WASD key capture. Sends /cmd?k=<key>&s=<1|0> on keydown/keyup.

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
<script>
// Pulse-driven: while key held, POST /cmd every 50ms.
// ESP auto-stops if no cmd in 600ms (deadman).
const keys = {w:false,a:false,s:false,d:false};
const log = document.getElementById('log');
const PULSE_MS = 50;

function send(k, down){
  fetch(`/cmd?k=${k}&s=${down?1:0}`).catch(e => log.textContent = 'cmd err: '+e);
}
function paint(){
  for (const k of ['w','a','s','d']) {
    document.getElementById('k'+k.toUpperCase()).className = keys[k]?'on':'';
  }
  log.textContent = `W=${+keys.w} A=${+keys.a} S=${+keys.s} D=${+keys.d}`;
}
setInterval(() => {
  for (const k of ['w','a','s','d']) if (keys[k]) send(k, true);
}, PULSE_MS);
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
